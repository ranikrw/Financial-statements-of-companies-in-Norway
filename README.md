# Financial statements of companies in Norway
This repository provides examples of how to handle the data described in:
Wahlstrøm, Ranik Raaen (2022). Financial statements of companies in Norway. arXiv:2203.12842. DOI: 10.48550/arXiv.2203.12842 URL: https://doi.org/10.48550/arXiv.2203.12842

The code in the file "main.py" loads the data, performs data filtering, prepares accounting variables, winsorizes the data, and provides an example of bankruptcy prediction. Functions used in "main.py" are found in the files in the folder "functions".

"results_of_bankruptcy_prediction_example.xlsx" presents results of the bankrupcty prediction example in the bottom of "main.py".

## Variable sets
The variable sets used in the code are derived from the following three studies:

Paraschiv, Florentina, Markus Schmid, and Ranik Raaen Wahlstrøm. 2021. “Bankruptcy Prediction of Privately Held SMEs Using Feature Selection Methods.” Working Paper, Norwegian University of Science and Technology and University of St. Gallen. https://doi.org/10.2139/ssrn.3911490

Altman, Edward I., and Gabriele Sabato. 2007. “Modelling Credit Risk for SMEs: Evidence from the U.S. Market.” Abacus 43 (3): 332–57. https://doi.org/10.1111/j.1467-6281.2007.00234.x

Altman, Edward I. 1968. “Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy.” The Journal of Finance 23 (4): 589–609. https://doi.org/10.2307/2978933

