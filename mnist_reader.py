def load_mnist(path, kind):
    import os
    import zipfile
    import pandas as pd

    """Load MNIST data from `path` """
    print("Loading %s Dataset..." %kind)
    zip_file_path = os.path.join(path, 'fashion-mnist_%s.csv.zip' %kind)
    # open the zip file
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        # extract the .csv file from the zip
        csv_file = zip_ref.namelist()[0]
        with zip_ref.open(csv_file) as file:
            # read the .csv file into a pandas DataFrame
            df = pd.read_csv(file)

    X = df.iloc[:, 1:].values/255.0
    y = df.iloc[:, 0].values
    return X, y