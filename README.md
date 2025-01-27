# ENM-Protein Corona Predictor

## Overview
The ENM-Protein Corona Predictor is a Python-based project designed to predict interactions between engineered nanomaterials (ENMs) and proteins. This tool leverages machine learning techniques to provide insights into protein corona formation, helping researchers understand key factors that govern these interactions. The project includes data preprocessing, model training, validation, and result visualization, making it a comprehensive solution for ENM-protein studies.

## Features
1. **Data Handling**:
   - Input data includes a database of ENM-protein interaction characteristics.
   - Supports custom datasets for predictions.
2. **Pipeline Execution**:
   - Automatically trains and tests a random forest classifier.
   - Allows recursive feature elimination (RFECV) and hyperparameter optimization.
3. **Performance Metrics**:
   - Generates validation metrics such as AUROC, recall, precision, and accuracy.
   - Outputs feature importance for better interpretability.
4. **Visualization**:
   - Includes ROC curve and Youden index visualizations.
   - Exports results in JSON and Excel formats for detailed analysis.
5. **Scalability**:
   - Allows multiple runs of the model to ensure robust predictions.

## Technologies Used
- **Programming Language**: Python
- **Libraries and Tools**:
  - Machine Learning: scikit-learn
  - Data Processing: NumPy, pandas
  - Visualization: Matplotlib
  - File Handling: JSON

## Code Highlights
The main components of the project include:
1. **Pipeline Functionality**:
   - `pipeline()`: Core function to train, evaluate, and validate the model.
2. **Classes and Utilities**:
   - `RandomForestClassifierWithCoef`: Custom random forest classifier with coefficient extraction.
   - `NpEncoder`: Handles JSON serialization for NumPy data types.
   - `data_utils`: Manages data preprocessing and feature selection.
3. **Validation Tools**:
   - `validation_utils`: Provides detailed validation metrics and performance tests.
   - `validator.y_randomization_test`: Ensures model reliability.

## Installation and Setup
1. Clone or download the repository.
2. Ensure Python (3.7 or later) is installed along with required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set the input database path in the `db.raw_data` attribute within the script.
4. Run the main script with the following arguments:
   ```bash
   python <script_name>.py <number_of_iterations> <output_file>
   ```

## How to Use
1. Prepare your input database file in `.csv` format.
2. Configure the script to use your custom dataset by setting `db.predict` to your file path.
3. Execute the script with desired iterations and output file name.
4. Review the results in the generated JSON and Excel files.

## Project Structure
- **Core Files**:
  - `data_utils.py`: Handles data preprocessing and splitting.
  - `predictor_utils.py`: Implements model training and optimization.
  - `validation_utils.py`: Provides metrics and validation methods.
  - `visualization_utils.py`: Generates visual insights.
  - `statistic_parser.py`: Converts JSON results into human-readable format.
- **Input Files**:
  - `database.csv`: Input dataset for model training and testing.
  - `_mask.txt`: RFECV mask file.
- **Output Files**:
  - JSON: Stores metrics and feature importances.
  - Excel: Contains classification information.

## Future Enhancements
1. Expand the model to support additional machine learning algorithms.
2. Implement a user-friendly GUI for non-technical users.
3. Add more detailed data visualization options.
4. Include automated hyperparameter tuning.

## License
This project is for educational and research purposes only. Redistribution and commercial use are not permitted without explicit permission.

## Contact
For questions or contributions, contact [MURALI SAI V] at [mv8039@srmist.edu.in].
