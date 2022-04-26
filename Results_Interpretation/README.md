# Intrepretation
This folder consists of all the files that are related to interpretation and observation of final results.

* Full.vs.Fraction_DS.ipynb is a python notebook to know all the classes present in full dataset and the classes retained in partial/experimental dataset

* Summary_Results.ipynb is a python notebook to find classification matrix for each class present in the partial/experimental dataset

* Bar_Charts.ipynb is a python notebook to generate bar charts corresponding to final results

* categories.csv is the csv file which has ground truth (y_true) a.k.a class label along with the class name

* iwildcam_split_test_seed_2_epoch_best_pred.csv is the csv file which corresponds to OOD test results y_pred along with y_true for strong data augmentation configuration

* iwildcam_split_test_seed_2_epoch_best_pred_with_aug.csv is the csv file which corresponds to OOD test results y_pred along with y_true for no data augmentation configuration

* classification_report_no_augmentation.csv is the csv file corresponding to classification report without data augmentation for the best model

* classification_report_with_augmentation.csv is the csv file corresponding to classification report with data augmentation for the best model
