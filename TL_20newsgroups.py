import sklearn
import random
from sklearn.datasets import fetch_20newsgroups, fetch_20newsgroups_vectorized
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from copy import deepcopy
import json
import pandas as pd
import numpy as np
from tensorlibrary.learning import CPKRR
from tensorlibrary.learning.transfer import CPKRR_LMPROJ, SVC_LMPROJ
from tensorlibrary.learning.features import pure_power_features, MMD, fourier_features
from imblearn.under_sampling import RandomUnderSampler

class TL20Newsgroups:
    """ Class to handle the 20 Newsgroups dataset for transfer learning tasks."""
    def __init__(self, categories=None, remove=('headers', 'footers', 'quotes'), subset='all', vectorized=False):
        """
        Initialize the TL20Newsgroups dataset.

        Args:
            categories (list, optional): List of categories to include. If None, all categories are included.
            remove (tuple, optional): Parts of the text to remove. Default is ('headers', 'footers', 'quotes').
            subset (str, optional): Subset of the dataset to load ('train', 'test', or 'all'). Default is 'all'.
            vectorized (bool, optional): If True, fetch the vectorized version of the dataset. Default is False.
        """
        if vectorized:
            raw_data = fetch_20newsgroups_vectorized(subset=subset, remove=remove, categories=categories)
        else:
            raw_data = fetch_20newsgroups(subset=subset, remove=remove, categories=categories)

        self.vectorized = vectorized
        self.data = raw_data.data
        self.labels = raw_data.target
        self.category_names = raw_data.target_names
        # extract subcategories (after dot in target_names)
        self.subcategories = self.category_names.copy()
        self.categories = [ name.split('.')[0] for name in self.category_names ]
        self.pipeline = None
        self.category_groups = self.create_category_groups()
        self.subcategory_groups = self.create_subcategory_groups()



    def create_pipeline(self, steps):
        """
        Create a machine learning pipeline with the specified vectorizer, scaler, and classifier.

        Args:
            vectorizer (object): A text vectorization object (e.g., CountVectorizer, TfidfVectorizer).
            scaler (object): A scaling object (e.g., StandardScaler).
            classifier (object): A classifier object (e.g., SVC, LogisticRegression).

        Returns:
            sklearn.pipeline.Pipeline: A scikit-learn pipeline object.
        """
        self.pipeline = Pipeline(steps)
        return self

    def create_category_groups(self):
        """
        Create a mapping of categories to their respective indices in the dataset.

        Returns:
            dict: A dictionary where keys are category names and values are lists of indices.
        """
        category_groups = {}
        for idx, label in enumerate(self.labels):
            category = self.category_names[label]
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(idx)

        self.category_groups = category_groups
        return category_groups

    def create_subcategory_groups(self):
        """
        Create a mapping of subcategories to their respective indices in the dataset.

        Returns:
            dict: A dictionary where keys are subcategory names and values are lists of indices.
        """
        subcategory_groups = {}
        for idx, label in enumerate(self.labels):
            subcategory = self.subcategories[label]
            if subcategory not in subcategory_groups:
                subcategory_groups[subcategory] = []
            subcategory_groups[subcategory].append(idx)

        self.subcategory_groups = subcategory_groups
        return subcategory_groups

    def vectorize_data(self, vectorizer, pca=None):
        """
        Vectorize the dataset using the provided vectorizer.

        Args:
            vectorizer (object): A text vectorization object (e.g., CountVectorizer, TfidfVectorizer).

        Returns:
            np.ndarray: The vectorized data.
        """
        if not self.vectorized:
            self.data = vectorizer.fit_transform(self.data)
            self.vectorized = True #
            if pca is not None:
                self.data = pca.fit_transform(self.data)
                self.data = self.data
            else:
                self.data = self.data.toarray()  # Convert sparse matrix to dense if PCA is not used
        return self

    def get_subcats_from_cat(self, cat):
        return [sub for sub in self.subcategory_groups if sub.startswith(cat)]

    def get_X_y(self, category_pos, category_neg):
        """
        For each category (positive and negative), randomly split its subcategories 50/50 into:
            - Source domain (D_s): for training
            - Target domain (D_t): for testing
        Uses all documents in selected subcategories.
        Returns:
            X_train, X_test, y_train, y_test, (subcategories used)
        """
        np.random.seed(RANDOM_STATE)
        pos_subcats = self.get_subcats_from_cat(category_pos)
        neg_subcats = self.get_subcats_from_cat(category_neg)

        if len(pos_subcats) < 2 or len(neg_subcats) < 2:
            raise ValueError(f"Not enough subcategories for {category_pos} or {category_neg}.")

        # Split subcategories 50/50
        np.random.shuffle(pos_subcats)
        np.random.shuffle(neg_subcats)

        half_pos = len(pos_subcats) // 2
        half_neg = len(neg_subcats) // 2

        pos_train_subcats = pos_subcats[:half_pos]
        pos_test_subcats = pos_subcats[half_pos:]

        neg_train_subcats = neg_subcats[:half_neg]
        neg_test_subcats = neg_subcats[half_neg:]

        def collect_docs(subcat_list, label):
            X = []
            y = []
            for subcat in subcat_list:
                indices = self.subcategory_groups[subcat]
                X.extend(self.data[indices,:])
                y.extend([label] * len(indices))
            X = np.asarray(X)
            y = np.asarray(y)
            return X, y

        X_train_pos, y_train_pos = collect_docs(pos_train_subcats, 1)
        X_train_neg, y_train_neg = collect_docs(neg_train_subcats, -1)
        X_test_pos, y_test_pos = collect_docs(pos_test_subcats, 1)
        X_test_neg, y_test_neg = collect_docs(neg_test_subcats, -1)

        X_train = np.concatenate([X_train_pos, X_train_neg])
        y_train = np.concatenate([y_train_pos, y_train_neg])

        X_test = np.concatenate([X_test_pos, X_test_neg])
        y_test = np.concatenate([y_test_pos, y_test_neg])

        # Shuffle the training and testing data
        train_indices = np.arange(len(y_train))
        np.random.shuffle(train_indices)
        X_train = X_train[train_indices]
        y_train = y_train[train_indices]
        test_indices = np.arange(len(y_test))
        np.random.shuffle(test_indices)
        X_test = X_test[test_indices]
        y_test = y_test[test_indices]

        # Create a dict

        subcats_used = {
            "pos_train_subcats": pos_train_subcats,
            "pos_test_subcats": pos_test_subcats,
            "neg_train_subcats": neg_train_subcats,
            "neg_test_subcats": neg_test_subcats
        }

        return X_train, X_test, y_train, y_test, subcats_used



    def crossvalidate(self, combinations, transductive=False, reduce_train=False, balance_train=False, out_of_sample=False, oos_percent=0.6, verbose=False, compute_mmd=False, mmd_fun=None):
        """
        Runs cross-validation across category pairs, returning predictions and metadata.

        Args:
            combinations (list of tuples): Each tuple is (category_pos, category_neg).
            transductive (bool): If True, uses transductive learning (x_target in fit).
            out_of_sample (bool): If True, uses out-of-sample set for testing, only applicable if transductive is True
            oos_percent (float): Percentage of out-of-sample data to use if out_of_sample is True.

        Returns:
            List[dict]: Results including true/pred labels and subcategory info.
        """
        results = []
        if transductive and verbose:
            print("Transductive learning mode enabled. Using x_target in fit.")

        for cat_pos, cat_neg in combinations:
            if verbose:
                print(f"Processing: {cat_pos} vs {cat_neg}")
            X_train, X_test, y_train, y_test, subcats_used = self.get_X_y(cat_pos, cat_neg)

            if reduce_train:
                # Reduce training set to 50% of the original size
                train_size = int(0.1 * len(y_train))
                indices = np.random.choice(len(y_train), train_size, replace=False)
                X_train = X_train[indices]
                y_train = y_train[indices]

            if balance_train:
                # Balance the training set using RandomUnderSampler
                rus = RandomUnderSampler(random_state=RANDOM_STATE)
                X_train, y_train = rus.fit_resample(X_train, y_train)
                rus_test = RandomUnderSampler(random_state=RANDOM_STATE)
                X_test, y_test = rus_test.fit_resample(X_test, y_test)


            if verbose:
                print(f"Training set size: {len(y_train)}, Test set size: {len(y_test)}")

            if not self.pipeline:
                raise ValueError("Pipeline not created. Use `create_pipeline()` first.")

            if compute_mmd:
                mmd = mmd_fun(X_train, X_test)
                if verbose:
                    print(f"MMD for {cat_pos} vs {cat_neg}: {mmd}")



            pipe = deepcopy(self.pipeline)
            if transductive:
                if out_of_sample:
                    X_target, y_target, X_test, y_test = train_test_split(X_test, y_test, test_size=oos_percent)
                else:
                    X_target = X_test
                mu = X_train.shape[0] + X_target.shape[0]
                pipe.set_params(clf__mu=mu)
                pipe.fit(X_train, y_train, clf__x_target=X_target)
            else:
                pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)

            result_entry = {
                'category_positive': cat_pos,
                'category_negative': cat_neg,
                'pos_train_subcats': subcats_used['pos_train_subcats'],
                'pos_test_subcats': subcats_used['pos_test_subcats'],
                'neg_train_subcats': subcats_used['neg_train_subcats'],
                'neg_test_subcats': subcats_used['neg_test_subcats'],
                'y_true': y_test,
                'y_pred': y_pred.tolist(),
                'mmd': mmd if compute_mmd else None
            }
            if verbose:
                print(f"Results for {cat_pos} vs {cat_neg}: {sklearn.metrics.accuracy_score(y_test, y_pred)}")

            results.append(result_entry)

        return results

    def save_results(self, results, path, fmt='json'):
        """
        Save results to a file.

        Args:
            results (list): Output of `crossvalidate`.
            path (str): File path to save.
            fmt (str): 'json' or 'csv'.
        """
        if fmt == 'json':
            with open(path, 'w') as f:
                json.dump(results, f, indent=4)
        elif fmt == 'csv':
            flat_rows = []
            for r in results:
                for true_label, pred_label in zip(r['y_true'], r['y_pred']):
                    flat_rows.append({
                        'category_positive': r['category_positive'],
                        'category_negative': r['category_negative'],
                        'pos_train_subcats': ','.join(r['pos_train_subcats']),
                        'pos_test_subcats': ','.join(r['pos_test_subcats']),
                        'neg_train_subcats': ','.join(r['neg_train_subcats']),
                        'neg_test_subcats': ','.join(r['neg_test_subcats']),
                        'y_true': true_label,
                        'y_pred': pred_label
                    })
            df = pd.DataFrame(flat_rows)
            df.to_csv(path, index=False)

    def print_results(self, results, show_predictions=False, max_samples=5):
        """
        Print a summary of cross-validation results.

        Args:
            results (list): Output of `crossvalidate`.
            show_predictions (bool): Whether to print predictions.
            max_samples (int): Number of predictions to show per task if enabled.
        """
        for r in results:
            print(f"\n=== {r['category_positive']} vs {r['category_negative']} ===")
            print(f"  Positive train subcats: {r['pos_train_subcats']}")
            print(f"  Positive test subcats : {r['pos_test_subcats']}")
            print(f"  Negative train subcats: {r['neg_train_subcats']}")
            print(f"  Negative test subcats : {r['neg_test_subcats']}")
            if show_predictions:
                print(f"  Predictions (first {max_samples}):")
                for i in range(min(max_samples, len(r['y_true']))):
                    print(f"    y_true: {r['y_true'][i]}, y_pred: {r['y_pred'][i]}")


if __name__ == "__main__":
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler
    from functools import partial
    from sklearn.svm import SVC
    RANDOM_STATE=10
    random.seed(RANDOM_STATE)
    categories = None  # load all
    pairs = [
        ("comp", "sci"),
        ("rec", "talk"),
        ("rec", "sci"),
        ("sci", "talk"),
        ("comp", "rec"),
        ("comp", "talk"),
    ]
    # gam = 5
    sig = 0.3
    M = 15
    Ld = 1.75
    R = 15
    reg_par = 1e-4
    C = 10.0
    gam =  np.sqrt(1/sig**2)
    # Compute MMD between source and target data
    feature_fun = partial(fourier_features, m=M, map_param=sig, Ld=Ld)
    COMPUTE_MMD = False
    REDUCE_TRAIN = False
    BALANCE_TRAIN = False

    # %% Model 1: CPKRR non-transductive
    model = TL20Newsgroups(categories=categories)
    # clf = CPKRR(
    #     feature_map='rbf',
    #     M=M,
    #     num_sweeps=15,
    #     map_param=sig,
    #     reg_par=reg_par,
    #     max_rank=R,
    #     Ld=Ld,
    #     random_init=True,
    #     train_loss_flag=False,
    #     random_state=RANDOM_STATE,
    #     class_weight=None,
    # )
    clf = SVC(
        C=C,
        kernel='rbf',
        gamma=gam,
    )
    scaler = MinMaxScaler((-0.5, 0.5)) #StandardScaler()
    vectorizer = TfidfVectorizer(stop_words='english')
    pca = PCA(n_components=20, random_state=RANDOM_STATE)
    # pca = None
    model.vectorize_data(vectorizer, pca=pca)

    steps = [
        ('scaler', scaler),
        ('clf', clf)
    ]

    model.create_pipeline(steps)

    mmd_fun = partial(MMD, feature_fun_source=feature_fun, feature_fun_target=feature_fun)

    results = model.crossvalidate(pairs, transductive=False, verbose=True, reduce_train=REDUCE_TRAIN, balance_train=BALANCE_TRAIN, compute_mmd=COMPUTE_MMD, mmd_fun=mmd_fun)
    # model.print_results(results, show_predictions=False)


    # %% MODEL 2: CPKRR LMPROJ transductive

    model = TL20Newsgroups(categories=categories)
    # clf = CPKRR_LMPROJ(
    #     feature_map='rbf',
    #     M=M,
    #     num_sweeps=15,
    #     map_param=sig,
    #     reg_par=reg_par,
    #     max_rank=R,
    #     random_init=True,
    #     Ld=Ld,
    #     train_loss_flag=False,
    #     mu="scale",
    #     random_state=RANDOM_STATE,
    #     # class_weight='balanced',
    # )

    clf = SVC_LMPROJ(
        kernel='rbf',
        C=C,
        mu="scale",
        gamma=gam,
        reg_par=reg_par,
        verbose=True
    )
    scaler =MinMaxScaler((-0.5, 0.5)) #StandardScaler()
    vectorizer = TfidfVectorizer(stop_words='english')
    pca = PCA(n_components=20, random_state=RANDOM_STATE)
    model.vectorize_data(vectorizer, pca=pca)

    steps = [
        ('scaler', scaler),
        ('clf', clf)
    ]

    model.create_pipeline(steps)

    mmd_fun = partial(MMD, feature_fun_source=feature_fun, feature_fun_target=feature_fun)

    results = model.crossvalidate(pairs, transductive=True, verbose=True, reduce_train=REDUCE_TRAIN, balance_train=BALANCE_TRAIN, compute_mmd=False, mmd_fun=mmd_fun)
    model.print_results(results, show_predictions=False)




