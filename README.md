# playground_series_gemstone_price_dataset
Tabular Regression with a Gemstone Price Dataset - Kaggle Competition 2023

## DVC run commands to generate the data
	- dvc run -n feat_engineering_processing -d src/feat_engineering.py -d data/train.csv -d data/test.csv -o data/train_prepared_feat_eng.csv -o data/test_prepared_feat_eng.csv python src/feat_engineering.py
	- dvc run -n feat_selection -d src/feat_selection.py -d data/train_prepared_feat_eng.csv -o data/selected_feat.csv python src/feat_selection.py
	- dvc run -n train-evaluate -d src/train_evaluate.py -d data/train_prepared_feat_eng.csv -d data/selected_feat.csv -o model/baseline_model.joblib -o model/minmax_scaler.joblib -M notes/report.json python src/train_evaluate.py


Some useful commands:

	- dvc dag <name_stage> 
	- dvc repro <name_stage>
	- dvc remote default <remote-tracker>