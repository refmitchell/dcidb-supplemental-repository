import pandas as pd
import numpy as np

"""
Quick robustness test for the assumption of uniform priors with
the cross-model analysis.
"""
if __name__ == "__main__":
    cmedf = pd.read_csv("cmedf.csv")
    cmedf["Unnamed: 0"] = [x.split(".")[0].split("-")[0] for x in cmedf["Unnamed: 0"]]
    cmedf.set_index("Unnamed: 0", inplace=True)

    # Df for maniuplation
    test_df = pd.DataFrame()

    # Clean up indices
    test_df["models"] = cmedf.index
    test_df.set_index("models", inplace=True)
    test_df["no_prior"] = cmedf["sum"]

    # For each model, see what adjustment in prior is required for them to win
    for m in test_df.index:
        print("Testing: {}".format(m))

        # Start with even priors for each model
        test_df["priors"] = np.ones(len(test_df)) * (1/len(test_df))
        test_df["log_priors"] = np.log(test_df["priors"])
        test_df["with_prior"] = test_df["no_prior"] + test_df["log_priors"]

        comparator = "BWSAw"
        if m == comparator: continue
        print("Pre-run: \n\n {}\n".format(test_df))

        while test_df.loc[m, "with_prior"] <= test_df.loc[comparator, "with_prior"]:
             # Increment target prior
             test_df.loc[m,"priors"] += 0.1

             # Normalise and update log priors
             test_df["priors"] = test_df["priors"].div(test_df["priors"].sum())
             test_df["log_priors"] = np.log(test_df["priors"])

             # Add new prior to even value.
             test_df["with_prior"] = test_df["no_prior"] + test_df["log_priors"]


        print("Model: {} requires prior of {} to beat {}".format(m, test_df.loc[m,"priors"], comparator))
        print("Dataframe: \n\n {}\n".format(test_df))
