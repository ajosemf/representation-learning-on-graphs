import argparse

def parameter_parser():
    """
    A method to parse up command line parameters.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path",
                        nargs = "?",
                        default = "../datasets",
	                help = "Dataset path. e.g. '../datasets'")

    parser.add_argument("--dataset_name",
                        nargs = "?",
                        required=True,
	                help = "Dataset name. e.g. 'pubmed'")

    parser.add_argument("--epochs",
                        type = int,
                        default = 130,
	                help = "Number of training epochs. Default is 130.")

    parser.add_argument("--early_stopping",
                    type = int,
                    default = 1000,
                help = "Tolerance for early stopping (# of epochs). Default is 1000.")

    parser.add_argument("--learning_rate",
                        nargs= "+",
                        type = float,
                        default = 0.01,
	                help = "Learning rate list. Default is 0.01.")

    parser.add_argument("--batch_size",
                        nargs= "+",
                        type = int,
                        default = 32,
	                help = "Batch size list. Default is 32.")

    parser.add_argument("--verbose",
                        action='store_true',
	                help = "Enable verbose output during training. Default is False.")
    
    return parser.parse_args()
