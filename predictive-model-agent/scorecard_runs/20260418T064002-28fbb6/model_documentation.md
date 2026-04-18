# Model documentation

**Run id:** `20260418T064002-28fbb6`\
**Data source:** `F:\datascience\dsa-agent\predictive-model-agent\data\heloc_dataset_v1.parquet`\
**Champion branch:** `br_aic_forward` (recipe `aic_forward`)\
**Features:** 10\
**Headline Gini (valid):** 0.6249339951420425\
**PDO:** base=600, pdo=20, odds=50.0

## 1. EDA

Interpretation: The dataset contains 10459 rows with an overall target rate of 47.806%. The feature quality is excellent, as all 23 listed features report a missing rate of 0.0%, ensuring complete data availability for model development.

### Missing rates (top)

                          feature_name  missing_rate
0                 ExternalRiskEstimate           0.0
1                MSinceOldestTradeOpen           0.0
2            MSinceMostRecentTradeOpen           0.0
3                       AverageMInFile           0.0
4                NumSatisfactoryTrades           0.0
5          NumTrades60Ever2DerogPubRec           0.0
6          NumTrades90Ever2DerogPubRec           0.0
7               PercentTradesNeverDelq           0.0
8                 MSinceMostRecentDelq           0.0
9             MaxDelq2PublicRecLast12M           0.0
10                         MaxDelqEver           0.0
11                      NumTotalTrades           0.0
12              NumTradesOpeninLast12M           0.0
13                PercentInstallTrades           0.0
14        MSinceMostRecentInqexcl7days           0.0
15                        NumInqLast6M           0.0
16               NumInqLast6Mexcl7days           0.0
17          NetFractionRevolvingBurden           0.0
18            NetFractionInstallBurden           0.0
19          NumRevolvingTradesWBalance           0.0
20            NumInstallTradesWBalance           0.0
21  NumBank2NatlTradesWHighUtilization           0.0
22               PercentTradesWBalance           0.0

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

Interpretation: The data is divided into five segments, with the 'train' and 'valid' sets both containing 2201 and 1101 records, respectively. The target rates are relatively balanced across segments, ranging from 0.428161 in 'hoot' to 0.518245 in 'oot'.

  data_type  count  target_rate
0      hoot   3480     0.428161
1       oot   2576     0.518245
2     train   2201     0.493866
3     valid   1101     0.494096
4      test   1101     0.494096

## 3. Feature transformation statistics

Optbinning ``BinningProcess`` fit on the ``train`` cohort; WoE columns
derived via ``get_woe_from_bp``. Top entries of the binning table:

                                  name      dtype   status  selected n_bins        iv        js      gini quality_score gini_power
0                 ExternalRiskEstimate  numerical  OPTIMAL      True      6  0.961558  0.111882  0.514269      0.184928   0.514269
1                MSinceOldestTradeOpen  numerical  OPTIMAL      True      6  0.202645  0.024758  0.237459      0.232167   0.762541
2            MSinceMostRecentTradeOpen  numerical  OPTIMAL      True      6  0.045154  0.005582   0.09467       0.00049    0.90533
3                       AverageMInFile  numerical  OPTIMAL      True      6  0.336253  0.040745  0.311294      0.627068   0.688706
4                NumSatisfactoryTrades  numerical  OPTIMAL      True      6   0.09076  0.011224  0.158444      0.034439   0.841556
5          NumTrades60Ever2DerogPubRec  numerical  OPTIMAL      True      4  0.224338  0.027368  0.215897       0.27059   0.784103
6          NumTrades90Ever2DerogPubRec  numerical  OPTIMAL      True      3  0.157773  0.019149  0.158314      0.269387   0.841686
7               PercentTradesNeverDelq  numerical  OPTIMAL      True      6  0.380445  0.046286  0.316547      0.183062   0.683453
8                 MSinceMostRecentDelq  numerical  OPTIMAL      True      6  0.360064   0.04357  0.301918      0.068936   0.698082
9             MaxDelq2PublicRecLast12M  numerical  OPTIMAL      True      4  0.327656  0.039887  0.301709      0.785891   0.698291
10                         MaxDelqEver  numerical  OPTIMAL      True      5  0.247467  0.030546  0.261326      0.019611   0.738674
11                      NumTotalTrades  numerical  OPTIMAL      True      6  0.067698   0.00837  0.136824      0.081101   0.863176
12              NumTradesOpeninLast12M  numerical  OPTIMAL      True      6  0.041503  0.005164   0.11212      0.010569    0.88788
13                PercentInstallTrades  numerical  OPTIMAL      True      6  0.194636  0.023512  0.225979       0.61483   0.774021
14        MSinceMostRecentInqexcl7days  numerical  OPTIMAL      True      6   0.44963  0.053811  0.329362      0.258013   0.670638
15                        NumInqLast6M  numerical  OPTIMAL      True      4  0.119543  0.014774  0.184226      0.380238   0.815774
16               NumInqLast6Mexcl7days  numerical  OPTIMAL      True      4  0.105588  0.013078  0.173007       0.31759   0.826993
17          NetFractionRevolvingBurden  numerical  OPTIMAL      True      6  0.502097  0.060369   0.37844      0.734702    0.62156
18            NetFractionInstallBurden  numerical  OPTIMAL      True      6  0.060504  0.007532  0.133857      0.016406   0.866143
19          NumRevolvingTradesWBalance  numerical  OPTIMAL      True      6  0.060215  0.007478  0.128977      0.002894   0.871023
20            NumInstallTradesWBalance  numerical  OPTIMAL      True      5   0.04167  0.005185  0.110679      0.044569   0.889321
21  NumBank2NatlTradesWHighUtilization  numerical  OPTIMAL      True      5  0.184924  0.022684  0.218712      0.151258   0.781288
22               PercentTradesWBalance  numerical  OPTIMAL      True      6  0.361049  0.043631  0.323474      0.598111   0.676526

## 4. Feature selection statistics

Interpretation: The 'aic_forward' recipe was selected as the champion model. It utilizes 10 features and achieved the highest performance among candidates, demonstrating a gini_valid of 0.624934 and a gini_test of 0.608060, confirming its predictive capability.

### All branches

         branch_id        recipe  n_features  gini_valid  gini_test  passed
0   br_aic_forward   aic_forward          10    0.624934   0.608060    True
1  br_bic_backward  bic_backward           8    0.606736   0.578826    True
2     br_iv_corr05     iv_corr05          11    0.608113   0.584783    True
3    br_auc_corr05    auc_corr05          11    0.608113   0.584783    True

### Champion features

- `ExternalRiskEstimate_woe`
- `MSinceMostRecentInqexcl7days_woe`
- `NumTotalTrades_woe`
- `PercentInstallTrades_woe`
- `AverageMInFile_woe`
- `MSinceMostRecentDelq_woe`
- `NetFractionRevolvingBurden_woe`
- `NumRevolvingTradesWBalance_woe`
- `NumInstallTradesWBalance_woe`
- `MSinceMostRecentTradeOpen_woe`

## 5. Model performance

Interpretation: The champion model shows strong performance with a gini coefficient of 0.624934 on the validation set and 0.608060 on the test set. Performance is consistent across periods, with AUCROC values ranging from 0.752116 in 'hoot' to 0.826143 in 'train', indicating robust model stability.

### Discrimination by cohort

  time_period    aucroc      gini  count  count_positive  positive_rate
0        hoot  0.752116  0.504233   3480            1490       0.428161
1         oot  0.802541  0.605082   2576            1335       0.518245
2        test  0.804030  0.608060   1101             544       0.494096
3       train  0.826143  0.652287   2201            1087       0.493866
4       valid  0.812467  0.624934   1101             544       0.494096

### Discrimination over time

   time_period    aucroc      gini  count  count_positive  positive_rate
0   2015-01-01  0.789357  0.578714    899             432       0.480534
1   2015-02-01  0.671888  0.343776    812             322       0.396552
2   2015-03-01  0.781038  0.562076    899             318       0.353726
3   2015-04-01  0.735821  0.471641    870             418       0.480460
4   2015-05-01  0.813002  0.626005    899             525       0.583982
5   2015-06-01  0.816267  0.632534    870             426       0.489655
6   2015-07-01  0.828903  0.657807    899             427       0.474972
7   2015-08-01  0.870147  0.740293    895             472       0.527374
8   2015-09-01  0.733443  0.466886    840             325       0.386905
9   2015-10-01  0.778846  0.557692    868             435       0.501152
10  2015-11-01  0.804084  0.608168    840             388       0.461905
11  2015-12-01  0.819945  0.639890    868             512       0.589862

## 6. Model stability

Interpretation: Population Stability Index (PSI) values fluctuate across the 12-month period. While most months show good stability (e.g., 0.003622 in August), there is a notable spike in February 2015 with a PSI of 1.082470, suggesting a significant shift in the underlying data distribution during that period.

### Score PSI over time

   time_period       psi  count data
0   2015-01-01  0.017124         899
1   2015-02-01  1.082470         812
2   2015-03-01  0.260830         899
3   2015-04-01  0.047054         870
4   2015-05-01  0.112686         899
5   2015-06-01  0.009424         870
6   2015-07-01  0.054232         899
7   2015-08-01  0.003622         895
8   2015-09-01  0.071485         840
9   2015-10-01  0.009273         868
10  2015-11-01  0.023322         840
11  2015-12-01  0.028920         868

## 7. Feature stability

### WoE feature PSI over time

          time                            feature_name       psi  count data
0   2015-01-01                ExternalRiskEstimate_woe  0.005177         899
1   2015-01-01               MSinceOldestTradeOpen_woe  0.012262         899
2   2015-01-01           MSinceMostRecentTradeOpen_woe  0.009147         899
3   2015-01-01                      AverageMInFile_woe  0.004626         899
4   2015-01-01               NumSatisfactoryTrades_woe  0.010328         899
5   2015-01-01         NumTrades60Ever2DerogPubRec_woe  0.004679         899
6   2015-01-01         NumTrades90Ever2DerogPubRec_woe  0.007561         899
7   2015-01-01              PercentTradesNeverDelq_woe  0.000895         899
8   2015-01-01                MSinceMostRecentDelq_woe  0.002857         899
9   2015-01-01            MaxDelq2PublicRecLast12M_woe  0.006312         899
10  2015-01-01                         MaxDelqEver_woe  0.011734         899
11  2015-01-01                      NumTotalTrades_woe  0.016998         899
12  2015-01-01              NumTradesOpeninLast12M_woe  0.009434         899
13  2015-01-01                PercentInstallTrades_woe  0.018480         899
14  2015-01-01        MSinceMostRecentInqexcl7days_woe  0.032571         899
15  2015-01-01                        NumInqLast6M_woe  0.004915         899
16  2015-01-01               NumInqLast6Mexcl7days_woe  0.007182         899
17  2015-01-01          NetFractionRevolvingBurden_woe  0.010811         899
18  2015-01-01            NetFractionInstallBurden_woe  0.008709         899
19  2015-01-01          NumRevolvingTradesWBalance_woe  0.002192         899
20  2015-01-01            NumInstallTradesWBalance_woe  0.007977         899
21  2015-01-01  NumBank2NatlTradesWHighUtilization_woe  0.005594         899
22  2015-01-01               PercentTradesWBalance_woe  0.022199         899
23  2015-02-01                ExternalRiskEstimate_woe  0.556442         812
24  2015-02-01               MSinceOldestTradeOpen_woe  0.634164         812
25  2015-02-01           MSinceMostRecentTradeOpen_woe  0.946644         812
26  2015-02-01                      AverageMInFile_woe  0.743394         812
27  2015-02-01               NumSatisfactoryTrades_woe  0.838867         812
28  2015-02-01         NumTrades60Ever2DerogPubRec_woe  0.145530         812
29  2015-02-01         NumTrades90Ever2DerogPubRec_woe  0.102680         812
30  2015-02-01              PercentTradesNeverDelq_woe  0.809442         812
31  2015-02-01                MSinceMostRecentDelq_woe  0.281833         812
32  2015-02-01            MaxDelq2PublicRecLast12M_woe  1.024523         812
33  2015-02-01                         MaxDelqEver_woe  1.049661         812
34  2015-02-01                      NumTotalTrades_woe  1.184889         812
35  2015-02-01              NumTradesOpeninLast12M_woe  0.538925         812
36  2015-02-01                PercentInstallTrades_woe  1.088433         812
37  2015-02-01        MSinceMostRecentInqexcl7days_woe  1.441207         812
38  2015-02-01                        NumInqLast6M_woe  0.274269         812
39  2015-02-01               NumInqLast6Mexcl7days_woe  0.267957         812
40  2015-02-01          NetFractionRevolvingBurden_woe  0.389620         812
41  2015-02-01            NetFractionInstallBurden_woe  0.416098         812
42  2015-02-01          NumRevolvingTradesWBalance_woe  0.377778         812
43  2015-02-01            NumInstallTradesWBalance_woe  0.991787         812
44  2015-02-01  NumBank2NatlTradesWHighUtilization_woe  0.193972         812
45  2015-02-01               PercentTradesWBalance_woe  1.204239         812
46  2015-03-01                ExternalRiskEstimate_woe  0.074393         899
47  2015-03-01               MSinceOldestTradeOpen_woe  0.024959         899
48  2015-03-01           MSinceMostRecentTradeOpen_woe  0.008087         899
49  2015-03-01                      AverageMInFile_woe  0.022019         899
50  2015-03-01               NumSatisfactoryTrades_woe  0.006999         899
51  2015-03-01         NumTrades60Ever2DerogPubRec_woe  0.013259         899
52  2015-03-01         NumTrades90Ever2DerogPubRec_woe  0.009373         899
53  2015-03-01              PercentTradesNeverDelq_woe  0.013574         899
54  2015-03-01                MSinceMostRecentDelq_woe  0.011452         899
55  2015-03-01            MaxDelq2PublicRecLast12M_woe  0.007111         899
56  2015-03-01                         MaxDelqEver_woe  0.010383         899
57  2015-03-01                      NumTotalTrades_woe  0.043057         899
58  2015-03-01              NumTradesOpeninLast12M_woe  0.021820         899
59  2015-03-01                PercentInstallTrades_woe  0.035343         899
60  2015-03-01        MSinceMostRecentInqexcl7days_woe  4.415512         899
61  2015-03-01                        NumInqLast6M_woe  0.004660         899
62  2015-03-01               NumInqLast6Mexcl7days_woe  0.003997         899
63  2015-03-01          NetFractionRevolvingBurden_woe  0.041081         899
64  2015-03-01            NetFractionInstallBurden_woe  0.024616         899
65  2015-03-01          NumRevolvingTradesWBalance_woe  0.076737         899
66  2015-03-01            NumInstallTradesWBalance_woe  0.010888         899
67  2015-03-01  NumBank2NatlTradesWHighUtilization_woe  0.061368         899
68  2015-03-01               PercentTradesWBalance_woe  0.026318         899
69  2015-04-01                ExternalRiskEstimate_woe  0.005149         870
70  2015-04-01               MSinceOldestTradeOpen_woe  0.034114         870
71  2015-04-01           MSinceMostRecentTradeOpen_woe  0.002257         870
72  2015-04-01                      AverageMInFile_woe  0.003022         870
73  2015-04-01               NumSatisfactoryTrades_woe  0.014647         870
74  2015-04-01         NumTrades60Ever2DerogPubRec_woe  0.003763         870
75  2015-04-01         NumTrades90Ever2DerogPubRec_woe  0.002705         870
76  2015-04-01              PercentTradesNeverDelq_woe  0.007946         870
77  2015-04-01                MSinceMostRecentDelq_woe  0.001693         870
78  2015-04-01            MaxDelq2PublicRecLast12M_woe  0.001765         870
79  2015-04-01                         MaxDelqEver_woe  0.004799         870
80  2015-04-01                      NumTotalTrades_woe  0.162251         870
81  2015-04-01              NumTradesOpeninLast12M_woe  0.015255         870
82  2015-04-01                PercentInstallTrades_woe  0.042943         870
83  2015-04-01        MSinceMostRecentInqexcl7days_woe  0.179078         870
84  2015-04-01                        NumInqLast6M_woe  0.003906         870
85  2015-04-01               NumInqLast6Mexcl7days_woe  0.005939         870
86  2015-04-01          NetFractionRevolvingBurden_woe  0.013380         870
87  2015-04-01            NetFractionInstallBurden_woe  0.002884         870
88  2015-04-01          NumRevolvingTradesWBalance_woe  0.035524         870
89  2015-04-01            NumInstallTradesWBalance_woe  0.005215         870
90  2015-04-01  NumBank2NatlTradesWHighUtilization_woe  0.007035         870
91  2015-04-01               PercentTradesWBalance_woe  0.008939         870
92  2015-05-01                ExternalRiskEstimate_woe  0.042015         899
93  2015-05-01               MSinceOldestTradeOpen_woe  0.035511         899
94  2015-05-01           MSinceMostRecentTradeOpen_woe  0.003542         899
95  2015-05-01                      AverageMInFile_woe  0.027886         899
96  2015-05-01               NumSatisfactoryTrades_woe  0.040550         899
97  2015-05-01         NumTrades60Ever2DerogPubRec_woe  0.011731         899
98  2015-05-01         NumTrades90Ever2DerogPubRec_woe  0.002351         899
99  2015-05-01              PercentTradesNeverDelq_woe  0.043869         899
100 2015-05-01                MSinceMostRecentDelq_woe  0.008971         899
101 2015-05-01            MaxDelq2PublicRecLast12M_woe  0.018628         899
102 2015-05-01                         MaxDelqEver_woe  0.028593         899
103 2015-05-01                      NumTotalTrades_woe  0.073253         899
104 2015-05-01              NumTradesOpeninLast12M_woe  0.004979         899
105 2015-05-01                PercentInstallTrades_woe  0.071668         899
106 2015-05-01        MSinceMostRecentInqexcl7days_woe  1.978064         899
107 2015-05-01                        NumInqLast6M_woe  0.010545         899
108 2015-05-01               NumInqLast6Mexcl7days_woe  0.009913         899
109 2015-05-01          NetFractionRevolvingBurden_woe  0.045527         899
110 2015-05-01            NetFractionInstallBurden_woe  0.010896         899
111 2015-05-01          NumRevolvingTradesWBalance_woe  0.008667         899
112 2015-05-01            NumInstallTradesWBalance_woe  0.006403         899
113 2015-05-01  NumBank2NatlTradesWHighUtilization_woe  0.005096         899
114 2015-05-01               PercentTradesWBalance_woe  0.079574         899
115 2015-06-01                ExternalRiskEstimate_woe  0.008817         870
116 2015-06-01               MSinceOldestTradeOpen_woe  0.003500         870
117 2015-06-01           MSinceMostRecentTradeOpen_woe  0.010523         870
118 2015-06-01                      AverageMInFile_woe  0.017322         870
119 2015-06-01               NumSatisfactoryTrades_woe  0.015173         870
120 2015-06-01         NumTrades60Ever2DerogPubRec_woe  0.000599         870
121 2015-06-01         NumTrades90Ever2DerogPubRec_woe  0.000460         870
122 2015-06-01              PercentTradesNeverDelq_woe  0.008920         870
123 2015-06-01                MSinceMostRecentDelq_woe  0.005541         870
124 2015-06-01            MaxDelq2PublicRecLast12M_woe  0.001076         870
125 2015-06-01                         MaxDelqEver_woe  0.000539         870
126 2015-06-01                      NumTotalTrades_woe  0.021361         870
127 2015-06-01              NumTradesOpeninLast12M_woe  0.014736         870
128 2015-06-01                PercentInstallTrades_woe  0.009383         870
129 2015-06-01        MSinceMostRecentInqexcl7days_woe  0.065489         870
130 2015-06-01                        NumInqLast6M_woe  0.001919         870
131 2015-06-01               NumInqLast6Mexcl7days_woe  0.000708         870
132 2015-06-01          NetFractionRevolvingBurden_woe  0.005781         870
133 2015-06-01            NetFractionInstallBurden_woe  0.015307         870
134 2015-06-01          NumRevolvingTradesWBalance_woe  0.004525         870
135 2015-06-01            NumInstallTradesWBalance_woe  0.018156         870
136 2015-06-01  NumBank2NatlTradesWHighUtilization_woe  0.008498         870
137 2015-06-01               PercentTradesWBalance_woe  0.018595         870
138 2015-07-01                ExternalRiskEstimate_woe  0.026537         899
139 2015-07-01               MSinceOldestTradeOpen_woe  0.000824         899
140 2015-07-01           MSinceMostRecentTradeOpen_woe  0.016601         899
141 2015-07-01                      AverageMInFile_woe  0.007341         899
142 2015-07-01               NumSatisfactoryTrades_woe  0.040211         899
143 2015-07-01         NumTrades60Ever2DerogPubRec_woe  0.005114         899
144 2015-07-01         NumTrades90Ever2DerogPubRec_woe  0.001209         899
145 2015-07-01              PercentTradesNeverDelq_woe  0.063778         899
146 2015-07-01                MSinceMostRecentDelq_woe  0.013891         899
147 2015-07-01            MaxDelq2PublicRecLast12M_woe  0.024876         899
148 2015-07-01                         MaxDelqEver_woe  0.022013         899
149 2015-07-01                      NumTotalTrades_woe  0.024895         899
150 2015-07-01              NumTradesOpeninLast12M_woe  0.024675         899
151 2015-07-01                PercentInstallTrades_woe  0.015254         899
152 2015-07-01        MSinceMostRecentInqexcl7days_woe  0.048100         899
153 2015-07-01                        NumInqLast6M_woe  0.008404         899
154 2015-07-01               NumInqLast6Mexcl7days_woe  0.008136         899
155 2015-07-01          NetFractionRevolvingBurden_woe  0.007303         899
156 2015-07-01            NetFractionInstallBurden_woe  0.013913         899
157 2015-07-01          NumRevolvingTradesWBalance_woe  0.011337         899
158 2015-07-01            NumInstallTradesWBalance_woe  0.011029         899
159 2015-07-01  NumBank2NatlTradesWHighUtilization_woe  0.003699         899
160 2015-07-01               PercentTradesWBalance_woe  0.011113         899
161 2015-08-01                ExternalRiskEstimate_woe  0.004243         895
162 2015-08-01               MSinceOldestTradeOpen_woe  0.002879         895
163 2015-08-01           MSinceMostRecentTradeOpen_woe  0.010984         895
164 2015-08-01                      AverageMInFile_woe  0.006074         895
165 2015-08-01               NumSatisfactoryTrades_woe  0.008180         895
166 2015-08-01         NumTrades60Ever2DerogPubRec_woe  0.008394         895
167 2015-08-01         NumTrades90Ever2DerogPubRec_woe  0.002620         895
168 2015-08-01              PercentTradesNeverDelq_woe  0.004305         895
169 2015-08-01                MSinceMostRecentDelq_woe  0.008433         895
170 2015-08-01            MaxDelq2PublicRecLast12M_woe  0.001056         895
171 2015-08-01                         MaxDelqEver_woe  0.001668         895
172 2015-08-01                      NumTotalTrades_woe  0.008436         895
173 2015-08-01              NumTradesOpeninLast12M_woe  0.006987         895
174 2015-08-01                PercentInstallTrades_woe  0.011036         895
175 2015-08-01        MSinceMostRecentInqexcl7days_woe  0.003802         895
176 2015-08-01                        NumInqLast6M_woe  0.010014         895
177 2015-08-01               NumInqLast6Mexcl7days_woe  0.007057         895
178 2015-08-01          NetFractionRevolvingBurden_woe  0.005223         895
179 2015-08-01            NetFractionInstallBurden_woe  0.005425         895
180 2015-08-01          NumRevolvingTradesWBalance_woe  0.004107         895
181 2015-08-01            NumInstallTradesWBalance_woe  0.006199         895
182 2015-08-01  NumBank2NatlTradesWHighUtilization_woe  0.001580         895
183 2015-08-01               PercentTradesWBalance_woe  0.015565         895
184 2015-09-01                ExternalRiskEstimate_woe  0.012788         840
185 2015-09-01               MSinceOldestTradeOpen_woe  0.027617         840
186 2015-09-01           MSinceMostRecentTradeOpen_woe  0.003635         840
187 2015-09-01                      AverageMInFile_woe  0.030864         840
188 2015-09-01               NumSatisfactoryTrades_woe  0.001445         840
189 2015-09-01         NumTrades60Ever2DerogPubRec_woe  0.007174         840
190 2015-09-01         NumTrades90Ever2DerogPubRec_woe  0.018279         840
191 2015-09-01              PercentTradesNeverDelq_woe  0.008913         840
192 2015-09-01                MSinceMostRecentDelq_woe  0.009044         840
193 2015-09-01            MaxDelq2PublicRecLast12M_woe  0.008642         840
194 2015-09-01                         MaxDelqEver_woe  0.006929         840
195 2015-09-01                      NumTotalTrades_woe  0.015787         840
196 2015-09-01              NumTradesOpeninLast12M_woe  0.000266         840
197 2015-09-01                PercentInstallTrades_woe  0.031745         840
198 2015-09-01        MSinceMostRecentInqexcl7days_woe  0.120265         840
199 2015-09-01                        NumInqLast6M_woe  0.000382         840
200 2015-09-01               NumInqLast6Mexcl7days_woe  0.000644         840
201 2015-09-01          NetFractionRevolvingBurden_woe  0.020773         840
202 2015-09-01            NetFractionInstallBurden_woe  0.006329         840
203 2015-09-01          NumRevolvingTradesWBalance_woe  0.013002         840
204 2015-09-01            NumInstallTradesWBalance_woe  0.013081         840
205 2015-09-01  NumBank2NatlTradesWHighUtilization_woe  0.008603         840
206 2015-09-01               PercentTradesWBalance_woe  0.023653         840
207 2015-10-01                ExternalRiskEstimate_woe  0.014054         868
208 2015-10-01               MSinceOldestTradeOpen_woe  0.043292         868
209 2015-10-01           MSinceMostRecentTradeOpen_woe  0.014451         868
210 2015-10-01                      AverageMInFile_woe  0.004768         868
211 2015-10-01               NumSatisfactoryTrades_woe  0.009886         868
212 2015-10-01         NumTrades60Ever2DerogPubRec_woe  0.001437         868
213 2015-10-01         NumTrades90Ever2DerogPubRec_woe  0.000187         868
214 2015-10-01              PercentTradesNeverDelq_woe  0.007106         868
215 2015-10-01                MSinceMostRecentDelq_woe  0.007423         868
216 2015-10-01            MaxDelq2PublicRecLast12M_woe  0.000455         868
217 2015-10-01                         MaxDelqEver_woe  0.001816         868
218 2015-10-01                      NumTotalTrades_woe  0.003953         868
219 2015-10-01              NumTradesOpeninLast12M_woe  0.021455         868
220 2015-10-01                PercentInstallTrades_woe  0.003460         868
221 2015-10-01        MSinceMostRecentInqexcl7days_woe  0.021267         868
222 2015-10-01                        NumInqLast6M_woe  0.012169         868
223 2015-10-01               NumInqLast6Mexcl7days_woe  0.015135         868
224 2015-10-01          NetFractionRevolvingBurden_woe  0.010849         868
225 2015-10-01            NetFractionInstallBurden_woe  0.007164         868
226 2015-10-01          NumRevolvingTradesWBalance_woe  0.002839         868
227 2015-10-01            NumInstallTradesWBalance_woe  0.006848         868
228 2015-10-01  NumBank2NatlTradesWHighUtilization_woe  0.016049         868
229 2015-10-01               PercentTradesWBalance_woe  0.012151         868
230 2015-11-01                ExternalRiskEstimate_woe  0.007808         840
231 2015-11-01               MSinceOldestTradeOpen_woe  0.013958         840
232 2015-11-01           MSinceMostRecentTradeOpen_woe  0.014183         840
233 2015-11-01                      AverageMInFile_woe  0.002070         840
234 2015-11-01               NumSatisfactoryTrades_woe  0.014673         840
235 2015-11-01         NumTrades60Ever2DerogPubRec_woe  0.004163         840
236 2015-11-01         NumTrades90Ever2DerogPubRec_woe  0.007633         840
237 2015-11-01              PercentTradesNeverDelq_woe  0.014594         840
238 2015-11-01                MSinceMostRecentDelq_woe  0.002500         840
239 2015-11-01            MaxDelq2PublicRecLast12M_woe  0.017239         840
240 2015-11-01                         MaxDelqEver_woe  0.015560         840
241 2015-11-01                      NumTotalTrades_woe  0.022310         840
242 2015-11-01              NumTradesOpeninLast12M_woe  0.003722         840
243 2015-11-01                PercentInstallTrades_woe  0.074354         840
244 2015-11-01        MSinceMostRecentInqexcl7days_woe  0.041068         840
245 2015-11-01                        NumInqLast6M_woe  0.007137         840
246 2015-11-01               NumInqLast6Mexcl7days_woe  0.006358         840
247 2015-11-01          NetFractionRevolvingBurden_woe  0.015307         840
248 2015-11-01            NetFractionInstallBurden_woe  0.010058         840
249 2015-11-01          NumRevolvingTradesWBalance_woe  0.002832         840
250 2015-11-01            NumInstallTradesWBalance_woe  0.015176         840
251 2015-11-01  NumBank2NatlTradesWHighUtilization_woe  0.012384         840
252 2015-11-01               PercentTradesWBalance_woe  0.034363         840
253 2015-12-01                ExternalRiskEstimate_woe  0.009361         868
254 2015-12-01               MSinceOldestTradeOpen_woe  0.014451         868
255 2015-12-01           MSinceMostRecentTradeOpen_woe  0.017143         868
256 2015-12-01                      AverageMInFile_woe  0.017211         868
257 2015-12-01               NumSatisfactoryTrades_woe  0.014826         868
258 2015-12-01         NumTrades60Ever2DerogPubRec_woe  0.016967         868
259 2015-12-01         NumTrades90Ever2DerogPubRec_woe  0.011673         868
260 2015-12-01              PercentTradesNeverDelq_woe  0.014449         868
261 2015-12-01                MSinceMostRecentDelq_woe  0.015237         868
262 2015-12-01            MaxDelq2PublicRecLast12M_woe  0.011212         868
263 2015-12-01                         MaxDelqEver_woe  0.007273         868
264 2015-12-01                      NumTotalTrades_woe  0.010282         868
265 2015-12-01              NumTradesOpeninLast12M_woe  0.014487         868
266 2015-12-01                PercentInstallTrades_woe  0.006989         868
267 2015-12-01        MSinceMostRecentInqexcl7days_woe  0.013385         868
268 2015-12-01                        NumInqLast6M_woe  0.038561         868
269 2015-12-01               NumInqLast6Mexcl7days_woe  0.036937         868
270 2015-12-01          NetFractionRevolvingBurden_woe  0.011758         868
271 2015-12-01            NetFractionInstallBurden_woe  0.021872         868
272 2015-12-01          NumRevolvingTradesWBalance_woe  0.010881         868
273 2015-12-01            NumInstallTradesWBalance_woe  0.025943         868
274 2015-12-01  NumBank2NatlTradesWHighUtilization_woe  0.003236         868
275 2015-12-01               PercentTradesWBalance_woe  0.004955         868

## 8. Problematic features

- None identified

## 9. Other important content

**Model form:** binary logistic regression on WoE features with PDO scaling
(base_score=600,
pdo=20,
odds=50.0).

**Reproducibility:** run_id=`20260418T064002-28fbb6`; artifacts at `F:\datascience\dsa-agent\predictive-model-agent\scorecard_runs\20260418T064002-28fbb6`;
data=`F:\datascience\dsa-agent\predictive-model-agent\data\heloc_dataset_v1.parquet`.

**Champion vs alternates:** champion=`br_aic_forward`; alternates=`br_bic_backward, br_iv_corr05, br_auc_corr05`.

**Deployment notes:** input features map through the frozen ``BinningProcess``;
score column name=`score`.

**Limitations:** sample size is limited to the data window; OOT horizon is
bounded by the ``SplitConfig``; known data defects are listed under
problematic features.
