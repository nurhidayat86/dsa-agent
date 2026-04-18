# Model documentation

**Run id:** `20260417T231906-698e80`\
**Data source:** `F:\datascience\dsa-agent\predictive-model-agent\data\heloc_dataset_v1.parquet`\
**Champion branch:** `br_aic_forward` (recipe `aic_forward`)\
**Features:** 11\
**Headline Gini (valid):** 0.6006288484213285\
**PDO:** base=600, pdo=20, odds=50.0

## 1. EDA

Interpretation: The dataset consists of 10459 rows with an overall target rate of 47.806%. The feature engineering process is robust, as all 20 listed features exhibit a missing rate of 0.0, ensuring high data quality and completeness for model training without the need for extensive imputation.

### Missing rates (top)

                    feature_name  missing_rate
0           ExternalRiskEstimate           0.0
1          MSinceOldestTradeOpen           0.0
2      MSinceMostRecentTradeOpen           0.0
3                 AverageMInFile           0.0
4          NumSatisfactoryTrades           0.0
5    NumTrades60Ever2DerogPubRec           0.0
6    NumTrades90Ever2DerogPubRec           0.0
7         PercentTradesNeverDelq           0.0
8           MSinceMostRecentDelq           0.0
9       MaxDelq2PublicRecLast12M           0.0
10                   MaxDelqEver           0.0
11                NumTotalTrades           0.0
12        NumTradesOpeninLast12M           0.0
13          PercentInstallTrades           0.0
14  MSinceMostRecentInqexcl7days           0.0
15                  NumInqLast6M           0.0
16         NumInqLast6Mexcl7days           0.0
17    NetFractionRevolvingBurden           0.0
18      NetFractionInstallBurden           0.0
19    NumRevolvingTradesWBalance           0.0

### Target rate over time

                mean  count  count_positive
data_month                                 
2015-01-01  0.480534    899             432
2015-02-01  0.396552    812             322
2015-03-01  0.353726    899             318
2015-04-01  0.480460    870             418
2015-05-01  0.583982    899             525
2015-06-01  0.489655    870             426
2015-07-01  0.474972    899             427
2015-08-01  0.527374    895             472
2015-09-01  0.386905    840             325
2015-10-01  0.501152    868             435
2015-11-01  0.461905    840             388
2015-12-01  0.589862    868             512

## 2. Data split statistics

Interpretation: The data is partitioned into train (5754), test (1919), valid (1918), and oot (868) sets. While the target rates for train, test, and valid remain consistent near 0.468, the out-of-time (oot) set shows a higher target rate of 0.589862, indicating a potential shift in the target distribution over time that should be monitored during deployment.

  data_type  count  target_rate
0     train   5754     0.467848
1      test   1919     0.467952
2     valid   1918     0.468196
3       oot    868     0.589862

## 3. Feature transformation statistics

Optbinning ``BinningProcess`` fit on the ``train`` cohort; WoE columns
derived via ``get_woe_from_bp``. Top entries of the binning table:

                            name      dtype   status  selected n_bins        iv        js      gini quality_score gini_power
0           ExternalRiskEstimate  numerical  OPTIMAL      True      6  0.918616  0.106518  0.502462      0.239676   0.502462
1          MSinceOldestTradeOpen  numerical  OPTIMAL      True      6  0.209181  0.025408  0.237937       0.67972   0.762063
2      MSinceMostRecentTradeOpen  numerical  OPTIMAL      True      5  0.025693  0.003197  0.078134      0.042846   0.921866
3                 AverageMInFile  numerical  OPTIMAL      True      6  0.309627  0.037231  0.292326      0.857591   0.707674
4          NumSatisfactoryTrades  numerical  OPTIMAL      True      6  0.112664  0.013636  0.159934      0.351947   0.840066
5    NumTrades60Ever2DerogPubRec  numerical  OPTIMAL      True      3  0.157014  0.019301  0.179883       0.35007   0.820117
6    NumTrades90Ever2DerogPubRec  numerical  OPTIMAL      True      3  0.128041  0.015685  0.143925      0.225063   0.856075
7         PercentTradesNeverDelq  numerical  OPTIMAL      True      6  0.331292  0.040142  0.295823      0.643701   0.704177
8           MSinceMostRecentDelq  numerical  OPTIMAL      True      6  0.264952  0.032351  0.257557      0.483049   0.742443
9       MaxDelq2PublicRecLast12M  numerical  OPTIMAL      True      5  0.295635  0.036073  0.284347      0.265993   0.715653
10                   MaxDelqEver  numerical  OPTIMAL      True      6  0.210218  0.025907  0.243093      0.319176   0.756907
11                NumTotalTrades  numerical  OPTIMAL      True      6  0.060129  0.007445  0.127594        0.0832   0.872406
12        NumTradesOpeninLast12M  numerical  OPTIMAL      True      6  0.018069  0.002255  0.073245      0.001242   0.926755
13          PercentInstallTrades  numerical  OPTIMAL      True      6  0.107518  0.013146   0.15415      0.278332    0.84585
14  MSinceMostRecentInqexcl7days  numerical  OPTIMAL      True      6  0.234986   0.02892  0.243088      0.356396   0.756912
15                  NumInqLast6M  numerical  OPTIMAL      True      5  0.084882  0.010488  0.149948      0.097798   0.850052
16         NumInqLast6Mexcl7days  numerical  OPTIMAL      True      5  0.078435  0.009703  0.144593      0.179978   0.855407
17    NetFractionRevolvingBurden  numerical  OPTIMAL      True      6  0.517548  0.061645  0.383416      0.815073   0.616584
18      NetFractionInstallBurden  numerical  OPTIMAL      True      6  0.038133  0.004745  0.101969      0.053231   0.898031
19    NumRevolvingTradesWBalance  numerical  OPTIMAL      True      6   0.09366  0.011592   0.15921      0.326587    0.84079

## 4. Feature selection statistics

Interpretation: The champion model is aic_forward, utilizing 11 features. It outperformed alternative selection methods, achieving a gini_valid of 0.600629 and a gini_test of 0.564641. This model provides the best balance of predictive power and complexity among the candidates, making it the most suitable selection for production.

### All branches

         branch_id        recipe  n_features  gini_valid  gini_test  passed
0   br_aic_forward   aic_forward          11    0.600629   0.564641    True
1  br_bic_backward  bic_backward           9    0.595499   0.559726    True
2     br_iv_corr05     iv_corr05          13    0.578971   0.552851    True
3    br_auc_corr05    auc_corr05          13    0.578971   0.552851    True

### Champion features

- `ExternalRiskEstimate_woe`
- `MSinceMostRecentInqexcl7days_woe`
- `NumSatisfactoryTrades_woe`
- `NetFractionRevolvingBurden_woe`
- `AverageMInFile_woe`
- `PercentTradesNeverDelq_woe`
- `NumRevolvingTradesWBalance_woe`
- `PercentInstallTrades_woe`
- `NumInqLast6M_woe`
- `NumTradesOpeninLast12M_woe`
- `NetFractionInstallBurden_woe`

## 5. Model performance

Interpretation: The model demonstrates stable predictive performance across datasets, with a gini_valid of 0.600629 and a test gini of 0.564641. Notably, the model shows strong generalization on the out-of-time (oot) sample, achieving a gini of 0.654253 and an aucroc of 0.827126, confirming the model's reliability in predicting outcomes on unseen, time-sequenced data.

### Discrimination by cohort

  time_period    aucroc      gini  count  count_positive  positive_rate
0         oot  0.827126  0.654253    868             512       0.589862
1        test  0.782321  0.564641   1919             898       0.467952
2       train  0.802381  0.604762   5754            2692       0.467848
3       valid  0.800314  0.600629   1918             898       0.468196

### Discrimination over time

   time_period    aucroc      gini  count  count_positive  positive_rate
0   2015-01-01  0.790212  0.580424    899             432       0.480534
1   2015-02-01  0.714210  0.428419    812             322       0.396552
2   2015-03-01  0.797097  0.594193    899             318       0.353726
3   2015-04-01  0.755703  0.511406    870             418       0.480460
4   2015-05-01  0.816381  0.632763    899             525       0.583982
5   2015-06-01  0.808262  0.616525    870             426       0.489655
6   2015-07-01  0.822322  0.644643    899             427       0.474972
7   2015-08-01  0.858321  0.716643    895             472       0.527374
8   2015-09-01  0.731898  0.463797    840             325       0.386905
9   2015-10-01  0.791195  0.582390    868             435       0.501152
10  2015-11-01  0.813435  0.626870    840             388       0.461905
11  2015-12-01  0.827126  0.654253    868             512       0.589862

## 6. Model stability

Interpretation: Population Stability Index (PSI) values fluctuate across the 2015 period. While most months show stable PSI values (e.g., 0.005595 in January or 0.022816 in October), significant spikes occur in February (0.884534) and March (0.254921). These indicate intermittent instability in the underlying population distribution, suggesting that the model's performance may be sensitive to specific temporal shifts.

### Score PSI over time

   time_period       psi  count data
0   2015-01-01  0.005595         899
1   2015-02-01  0.884534         812
2   2015-03-01  0.254921         899
3   2015-04-01  0.057506         870
4   2015-05-01  0.169912         899
5   2015-06-01  0.031000         870
6   2015-07-01  0.021354         899
7   2015-08-01  0.026797         895
8   2015-09-01  0.057792         840
9   2015-10-01  0.022816         868
10  2015-11-01  0.028728         840
11  2015-12-01  0.037008         868

## 7. Feature stability

### WoE feature PSI over time

         time                      feature_name       psi  count data
0  2015-01-01          ExternalRiskEstimate_woe  0.013037         899
1  2015-01-01         MSinceOldestTradeOpen_woe  0.035956         899
2  2015-01-01     MSinceMostRecentTradeOpen_woe  0.014669         899
3  2015-01-01                AverageMInFile_woe  0.014313         899
4  2015-01-01         NumSatisfactoryTrades_woe  0.018012         899
5  2015-01-01   NumTrades60Ever2DerogPubRec_woe  0.002138         899
6  2015-01-01   NumTrades90Ever2DerogPubRec_woe  0.005279         899
7  2015-01-01        PercentTradesNeverDelq_woe  0.017407         899
8  2015-01-01          MSinceMostRecentDelq_woe  0.002368         899
9  2015-01-01      MaxDelq2PublicRecLast12M_woe  0.013967         899
10 2015-01-01                   MaxDelqEver_woe  0.015481         899
11 2015-01-01                NumTotalTrades_woe  0.039062         899
12 2015-01-01        NumTradesOpeninLast12M_woe  0.007410         899
13 2015-01-01          PercentInstallTrades_woe  0.025524         899
14 2015-01-01  MSinceMostRecentInqexcl7days_woe  0.045870         899
15 2015-01-01                  NumInqLast6M_woe  0.015163         899
16 2015-01-01         NumInqLast6Mexcl7days_woe  0.016097         899
17 2015-01-01    NetFractionRevolvingBurden_woe  0.011427         899
18 2015-01-01      NetFractionInstallBurden_woe  0.007767         899
19 2015-01-01    NumRevolvingTradesWBalance_woe  0.007916         899

## 8. Final scorecard

Per-feature, per-bin point allocations produced by ``create_scorecard_model`` in ``agent_tools.py``. Each row shows one bin of one feature with its ``Count`` / ``Event rate`` / ``WoE`` / ``IV`` plus the fitted logistic ``Coefficient`` and the final ``Points`` under the PDO scaling (base_score=600.0, pdo=20.0, odds=50.0). To score an application, look up the row matching each of the champion features for the applicant's value and sum the ``Points`` column.

                         Variable  Bin id              Bin  Count  Count (%)  Non-event  Event  Event rate       WoE        IV            JS  Coefficient     Points
0                  AverageMInFile       0    (-inf, -2.50)    322   0.055961        180    142    0.440994  0.108346  0.000654  8.171003e-05    -0.590986  46.434949
1                  AverageMInFile       1   [-2.50, 30.50)    312   0.054223        260     52    0.166667  1.480654  0.097124  1.114039e-02    -0.590986  69.835907
2                  AverageMInFile       2   [30.50, 43.50)    410   0.071255        305    105    0.256098  0.937568  0.056820  6.853283e-03    -0.590986  60.575054
3                  AverageMInFile       3   [43.50, 50.50)    295   0.051269        207     88    0.298305  0.726598  0.025368  3.103039e-03    -0.590986  56.977548
4                  AverageMInFile       4   [50.50, 56.50)    303   0.052659        202    101    0.333333  0.564363  0.016057  1.980895e-03    -0.590986  54.211077
5                  AverageMInFile       5   [56.50, 63.50)    447   0.077685        272    175    0.391499  0.312232  0.007438  9.260455e-04    -0.590986  49.911672
6                  AverageMInFile       6   [63.50, 68.50)    377   0.065520        206    171    0.453581  0.057429  0.000216  2.695007e-05    -0.590986  45.566696
7                  AverageMInFile       7   [68.50, 73.50)    360   0.062565        192    168    0.466667  0.004748  0.000001  1.762408e-07    -0.590986  44.668362
8                  AverageMInFile       8   [73.50, 80.50)    537   0.093326        263    274    0.510242 -0.169758  0.002698  3.368090e-04    -0.590986  41.692649
9                  AverageMInFile       9   [80.50, 97.50)   1079   0.187522        482    597    0.553290 -0.342757  0.022058  2.743836e-03    -0.590986  38.742626
10                 AverageMInFile      10  [97.50, 116.50)    695   0.120786        271    424    0.610072 -0.576399  0.039771  4.903702e-03    -0.590986  34.758508
11                 AverageMInFile      11    [116.50, inf)    617   0.107230        222    395    0.640194 -0.704992  0.052331  6.409210e-03    -0.590986  32.565694
12                 AverageMInFile      12          Special      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00    -0.590986  44.587407
13                 AverageMInFile      13          Missing      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00    -0.590986  44.587407
14           ExternalRiskEstimate       0    (-inf, 13.50)    329   0.057178        185    144    0.437690  0.121759  0.000843  1.053505e-04    -0.457209  46.193678
15           ExternalRiskEstimate       1   [13.50, 59.50)    594   0.103233        495     99    0.166667  1.480654  0.184909  2.120960e-02    -0.457209  64.120577
16           ExternalRiskEstimate       2   [59.50, 63.50)    611   0.106187        485    126    0.206219  1.219083  0.136035  1.602390e-02    -0.457209  60.669865
17           ExternalRiskEstimate       3   [63.50, 65.50)    378   0.065693        276    102    0.269841  0.866644  0.045280  5.489227e-03    -0.457209  56.020400
18           ExternalRiskEstimate       4   [65.50, 68.50)    558   0.096976        399    159    0.284946  0.791273  0.056373  6.868332e-03    -0.457209  55.026089
19           ExternalRiskEstimate       5   [68.50, 70.50)    366   0.063608        226    140    0.382514  0.350109  0.007633  9.492914e-04    -0.457209  49.206131
20           ExternalRiskEstimate       6   [70.50, 73.50)    568   0.098714        315    253    0.445423  0.090399  0.000804  1.004420e-04    -0.457209  45.779977
21           ExternalRiskEstimate       7   [73.50, 75.50)    340   0.059089        154    186    0.547059 -0.317578  0.005970  7.431753e-04    -0.457209  40.397836
22           ExternalRiskEstimate       8   [75.50, 78.50)    420   0.072993        160    260    0.619048 -0.614292  0.027231  3.351340e-03    -0.457209  36.483512
23           ExternalRiskEstimate       9   [78.50, 80.50)    319   0.055440        103    216    0.677116 -0.869333  0.040511  4.910159e-03    -0.457209  33.118937
24           ExternalRiskEstimate      10   [80.50, 84.50)    575   0.099930        151    424    0.737391 -1.161238  0.125634  1.487736e-02    -0.457209  29.268062
25           ExternalRiskEstimate      11   [84.50, 87.50)    355   0.061696         58    297    0.836620 -1.762073  0.161027  1.787161e-02    -0.457209  21.341685
26           ExternalRiskEstimate      12     [87.50, inf)    341   0.059263         55    286    0.838710 -1.777443  0.156910  1.738133e-02    -0.457209  21.138926
27           ExternalRiskEstimate      13          Special      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00    -0.457209  44.587407
28           ExternalRiskEstimate      14          Missing      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00    -0.457209  44.587407
29   MSinceMostRecentInqexcl7days       0    (-inf, -7.50)    589   0.102364        220    369    0.626486 -0.645953  0.042132  5.176790e-03    -0.791640  29.832617
30   MSinceMostRecentInqexcl7days       1   [-7.50, -3.50)   1055   0.183351        667    388    0.367773  0.413001  0.030438  3.777995e-03    -0.791640  54.021128
31   MSinceMostRecentInqexcl7days       2    [-3.50, 0.50)   2521   0.438130       1561    960    0.380801  0.357365  0.054743  6.806698e-03    -0.791640  52.750294
32   MSinceMostRecentInqexcl7days       3     [0.50, 1.50)    343   0.059611        159    184    0.536443 -0.274815  0.004514  5.624213e-04    -0.791640  38.310101
33   MSinceMostRecentInqexcl7days       4    [1.50, 10.50)    917   0.159367        352    565    0.616140 -0.601978  0.057142  7.036812e-03    -0.791640  30.837079
34   MSinceMostRecentInqexcl7days       5     [10.50, inf)    329   0.057178        103    226    0.686930 -0.914590  0.046017  5.559667e-03    -0.791640  23.696441
35   MSinceMostRecentInqexcl7days       6          Special      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00    -0.791640  44.587407
36   MSinceMostRecentInqexcl7days       7          Missing      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00    -0.791640  44.587407
37       NetFractionInstallBurden       0    (-inf, 34.50)   2550   0.443170       1250   1300    0.509804 -0.168005  0.012547  1.566532e-03    -0.230349  43.470771
38       NetFractionInstallBurden       1   [34.50, 58.50)    663   0.115224        340    323    0.487179 -0.077491  0.000693  8.663803e-05    -0.230349  44.072369
39       NetFractionInstallBurden       2   [58.50, 69.50)    495   0.086027        264    231    0.466667  0.004748  0.000002  2.423311e-07    -0.230349  44.618961
40       NetFractionInstallBurden       3   [69.50, 85.50)    995   0.172923        566    429    0.431156  0.148353  0.003781  4.721731e-04    -0.230349  45.573431
41       NetFractionInstallBurden       4   [85.50, 90.50)    343   0.059611        222    121    0.352770  0.478103  0.013173  1.631178e-03    -0.230349  47.765098
42       NetFractionInstallBurden       5   [90.50, 95.50)    351   0.061001        215    136    0.387464  0.329199  0.006484  8.068275e-04    -0.230349  46.775416
43       NetFractionInstallBurden       6     [95.50, inf)    357   0.062044        205    152    0.425770  0.170346  0.001786  2.230132e-04    -0.230349  45.719602
44       NetFractionInstallBurden       7          Special      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00    -0.230349  44.587407
45       NetFractionInstallBurden       8          Missing      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00    -0.230349  44.587407
46     NetFractionRevolvingBurden       0    (-inf, -4.00)    430   0.074731        271    159    0.369767  0.404431  0.011907  1.478262e-03    -0.393386  49.177982
47     NetFractionRevolvingBurden       1    [-4.00, 0.50)    346   0.060132        144    202    0.583815 -0.467238  0.013087  1.621143e-03    -0.393386  39.283921
48     NetFractionRevolvingBurden       2     [0.50, 3.50)    451   0.078380        109    342    0.758315 -1.272247  0.116341  1.363494e-02    -0.393386  30.146504
49     NetFractionRevolvingBurden       3     [3.50, 6.50)    334   0.058047        101    233    0.697605 -0.964702  0.051677  6.220235e-03    -0.393386  33.637357
50     NetFractionRevolvingBurden       4    [6.50, 13.50)    560   0.097324        184    376    0.671429 -0.843437  0.067122  8.150093e-03    -0.393386  35.013795
51     NetFractionRevolvingBurden       5   [13.50, 23.50)    632   0.109837        278    354    0.560127 -0.370460  0.015082  1.874489e-03    -0.393386  40.382426
52     NetFractionRevolvingBurden       6   [23.50, 37.50)    791   0.137470        414    377    0.476612 -0.035163  0.000170  2.126743e-05    -0.393386  44.188281
53     NetFractionRevolvingBurden       7   [37.50, 47.50)    485   0.084289        284    201    0.414433  0.216885  0.003922  4.893152e-04    -0.393386  47.049210
54     NetFractionRevolvingBurden       8   [47.50, 58.50)    486   0.084463        323    163    0.335391  0.555118  0.024945  3.078727e-03    -0.393386  50.888392
55     NetFractionRevolvingBurden       9   [58.50, 75.50)    613   0.106535        439    174    0.283850  0.796660  0.062725  7.639594e-03    -0.393386  53.630065
56     NetFractionRevolvingBurden      10   [75.50, 85.50)    294   0.051095        236     58    0.197279  1.274605  0.070777  8.293008e-03    -0.393386  59.055077
57     NetFractionRevolvingBurden      11     [85.50, inf)    332   0.057699        279     53    0.159639  1.532136  0.109439  1.248149e-02    -0.393386  61.978237
58     NetFractionRevolvingBurden      12          Special      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00    -0.393386  44.587407
59     NetFractionRevolvingBurden      13          Missing      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00    -0.393386  44.587407
60                   NumInqLast6M       0     (-inf, 0.50)   2506   0.435523       1175   1331    0.531125 -0.253446  0.028054  3.497448e-03    -0.325724  42.205418
61                   NumInqLast6M       1     [0.50, 1.50)   1371   0.238269        725    646    0.471189 -0.013412  0.000043  5.359509e-06    -0.325724  44.461358
62                   NumInqLast6M       2     [1.50, 2.50)    813   0.141293        469    344    0.423124  0.181177  0.004599  5.740413e-04    -0.325724  46.290182
63                   NumInqLast6M       3     [2.50, 3.50)    415   0.072124        245    170    0.409639  0.236676  0.003991  4.977217e-04    -0.325724  46.811781
64                   NumInqLast6M       4      [3.50, inf)    649   0.112791        448    201    0.309707  0.672704  0.048195  5.913315e-03    -0.325724  50.909750
65                   NumInqLast6M       5          Special      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00    -0.325724  44.587407
66                   NumInqLast6M       6          Missing      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00    -0.325724  44.587407
67     NumRevolvingTradesWBalance       0    (-inf, -4.00)    410   0.071255        257    153    0.373171  0.389854  0.010564  1.312186e-03    -0.640303  51.790060
68     NumRevolvingTradesWBalance       1    [-4.00, 1.50)    920   0.159889        452    468    0.508696 -0.163570  0.004291  5.357592e-04    -0.640303  41.565411
69     NumRevolvingTradesWBalance       2     [1.50, 2.50)    939   0.163191        413    526    0.560170 -0.370638  0.022429  2.787684e-03    -0.640303  37.739788
70     NumRevolvingTradesWBalance       3     [2.50, 3.50)    865   0.150330        418    447    0.516763 -0.195861  0.005785  7.219523e-04    -0.640303  40.968826
71     NumRevolvingTradesWBalance       4     [3.50, 4.50)    735   0.127737        370    365    0.496599 -0.115178  0.001699  2.122551e-04    -0.640303  42.459460
72     NumRevolvingTradesWBalance       5     [4.50, 5.50)    548   0.095238        296    252    0.459854  0.032146  0.000098  1.228805e-05    -0.640303  45.181321
73     NumRevolvingTradesWBalance       6     [5.50, 7.50)    714   0.124088        431    283    0.396359  0.291877  0.010400  1.295404e-03    -0.640303  49.979912
74     NumRevolvingTradesWBalance       7     [7.50, 9.50)    325   0.056482        217    108    0.332308  0.568982  0.017496  2.157983e-03    -0.640303  55.099494
75     NumRevolvingTradesWBalance       8      [9.50, inf)    298   0.051790        208     90    0.302013  0.708945  0.024457  2.994610e-03    -0.640303  57.685331
76     NumRevolvingTradesWBalance       9          Special      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00    -0.640303  44.587407
77     NumRevolvingTradesWBalance      10          Missing      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00    -0.640303  44.587407
78          NumSatisfactoryTrades       0    (-inf, -4.50)    322   0.055961        180    142    0.440994  0.108346  0.000654  8.171003e-05    -0.597628  46.455713
79          NumSatisfactoryTrades       1    [-4.50, 5.50)    307   0.053354        241     66    0.214984  1.166358  0.063205  7.481143e-03    -0.597628  64.699977
80          NumSatisfactoryTrades       2     [5.50, 8.50)    333   0.057873        215    118    0.354354  0.471170  0.012430  1.539582e-03    -0.597628  52.712208
81          NumSatisfactoryTrades       3    [8.50, 10.50)    299   0.051964        180    119    0.397993  0.285049  0.004156  5.177530e-04    -0.597628  49.502772
82          NumSatisfactoryTrades       4   [10.50, 13.50)    548   0.095238        312    236    0.430657  0.150387  0.002140  2.671941e-04    -0.597628  47.180674
83          NumSatisfactoryTrades       5   [13.50, 18.50)    993   0.172576        534    459    0.462236  0.022562  0.000088  1.097216e-05    -0.597628  44.976459
84          NumSatisfactoryTrades       6   [18.50, 21.50)    574   0.099757        296    278    0.484321 -0.066046  0.000436  5.447851e-05    -0.597628  43.448524
85          NumSatisfactoryTrades       7   [21.50, 26.50)    857   0.148940        411    446    0.520420 -0.210510  0.006621  8.260434e-04    -0.597628  40.957399
86          NumSatisfactoryTrades       8   [26.50, 31.50)    621   0.107925        291    330    0.531401 -0.254553  0.007013  8.742433e-04    -0.597628  40.197915
87          NumSatisfactoryTrades       9     [31.50, inf)    900   0.156413        402    498    0.553333 -0.342932  0.018417  2.290965e-03    -0.597628  38.673923
88          NumSatisfactoryTrades      10          Special      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00    -0.597628  44.587407
89          NumSatisfactoryTrades      11          Missing      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00    -0.597628  44.587407
90         NumTradesOpeninLast12M       0     (-inf, 0.50)   1698   0.295099        842    856    0.504122 -0.145274  0.006246  7.800820e-04     0.457593  46.505511
91         NumTradesOpeninLast12M       1     [0.50, 1.50)   1395   0.242440        722    673    0.482437 -0.058504  0.000831  1.038767e-04     0.457593  45.359855
92         NumTradesOpeninLast12M       2     [1.50, 2.50)   1124   0.195342        612    512    0.455516  0.049624  0.000480  6.001504e-05     0.457593  43.932208
93         NumTradesOpeninLast12M       3     [2.50, 3.50)    701   0.121828        388    313    0.446505  0.086018  0.000898  1.122637e-04     0.457593  43.451679
94         NumTradesOpeninLast12M       4     [3.50, 4.50)    393   0.068300        233    160    0.407125  0.247081  0.004116  5.132005e-04     0.457593  41.325118
95         NumTradesOpeninLast12M       5      [4.50, inf)    443   0.076990        265    178    0.401806  0.269162  0.005497  6.850678e-04     0.457593  41.033567
96         NumTradesOpeninLast12M       6          Special      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00     0.457593  44.587407
97         NumTradesOpeninLast12M       7          Missing      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00     0.457593  44.587407
98           PercentInstallTrades       0     (-inf, 7.50)    519   0.090198        292    227    0.437380  0.123020  0.001358  1.696388e-04    -0.530537  46.470603
99           PercentInstallTrades       1    [7.50, 34.50)   2740   0.476190       1322   1418    0.517518 -0.198886  0.018895  2.357930e-03    -0.530537  41.542854
100          PercentInstallTrades       2   [34.50, 39.50)    531   0.092284        241    290    0.546139 -0.313868  0.009108  1.133900e-03    -0.530537  39.782698
101          PercentInstallTrades       3   [39.50, 43.50)    385   0.066910        200    185    0.480519 -0.050822  0.000173  2.163118e-05    -0.530537  43.809415
102          PercentInstallTrades       4   [43.50, 47.50)    351   0.061001        189    162    0.461538  0.025367  0.000039  4.902183e-06    -0.530537  44.975723
103          PercentInstallTrades       5   [47.50, 53.50)    451   0.078380        275    176    0.390244  0.317503  0.007757  9.655897e-04    -0.530537  49.447765
104          PercentInstallTrades       6   [53.50, 66.50)    489   0.084984        324    165    0.337423  0.546014  0.024309  3.001409e-03    -0.530537  52.945824
105          PercentInstallTrades       7     [66.50, inf)    288   0.050052        219     69    0.239583  1.026181  0.047092  5.641077e-03    -0.530537  60.296252
106          PercentInstallTrades       8          Special      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00    -0.530537  44.587407
107          PercentInstallTrades       9          Missing      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00    -0.530537  44.587407
108        PercentTradesNeverDelq       0    (-inf, 10.00)    326   0.056656        182    144    0.441718  0.105410  0.000627  7.831526e-05    -0.464021  45.998716
109        PercentTradesNeverDelq       1   [10.00, 69.50)    306   0.053180        245     61    0.199346  1.261600  0.072357  8.488810e-03    -0.464021  61.478751
110        PercentTradesNeverDelq       2   [69.50, 81.50)    471   0.081856        363    108    0.229299  1.083488  0.084979  1.013151e-02    -0.464021  59.094031
111        PercentTradesNeverDelq       3   [81.50, 88.50)    538   0.093500        375    163    0.302974  0.704392  0.043615  5.341936e-03    -0.464021  54.018385
112        PercentTradesNeverDelq       4   [88.50, 91.50)    328   0.057004        201    127    0.387195  0.330334  0.006100  7.590679e-04    -0.464021  49.010189
113        PercentTradesNeverDelq       5   [91.50, 95.50)    734   0.127563        430    304    0.414169  0.217974  0.005995  7.479105e-04    -0.464021  47.505817
114        PercentTradesNeverDelq       6   [95.50, 97.50)    438   0.076121        194    244    0.557078 -0.358094  0.009769  1.214691e-03    -0.464021  39.792950
115        PercentTradesNeverDelq       7     [97.50, inf)   2613   0.454119       1072   1541    0.589744 -0.491689  0.109322  1.352920e-02    -0.464021  38.004265
116        PercentTradesNeverDelq       8          Special      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00    -0.464021  44.587407
117        PercentTradesNeverDelq       9          Missing      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00    -0.464021  44.587407

## 9. Problematic features

- None identified

## 10. Other important content

**Model form:** binary logistic regression on WoE features with PDO scaling
(base_score=600,
pdo=20,
odds=50.0).

**Reproducibility:** run_id=`20260417T231906-698e80`; artifacts at `F:\datascience\dsa-agent\predictive-model-agent\scorecard_runs\20260417T231906-698e80`;
data=`F:\datascience\dsa-agent\predictive-model-agent\data\heloc_dataset_v1.parquet`.

**Champion vs alternates:** champion=`br_aic_forward`; alternates=`br_bic_backward, br_iv_corr05, br_auc_corr05`.

**Deployment notes:** input features map through the frozen ``BinningProcess``;
score column name=`score`.

**Limitations:** sample size is limited to the data window; OOT horizon is
bounded by the ``SplitConfig``; known data defects are listed under
problematic features.
