# Model documentation

**Run id:** `20260418T082224-968a7b`\
**Data source:** `F:\datascience\dsa-agent\predictive-model-agent\data\heloc_dataset_v1.parquet`\
**Champion branch:** `br_aic_forward` (recipe `aic_forward`)\
**Features:** 9\
**Headline Gini (valid):** 0.599206226483171\
**PDO:** base=600, pdo=20, odds=50.0

## 1. EDA

Interpretation: The dataset consists of 10459 rows with a global target rate of 47.806%. All 23 listed features exhibit a missing rate of 0.0%, indicating a clean, complete dataset suitable for modeling without the need for imputation.

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

Interpretation: The data is divided into five segments, with target rates ranging from 42.8161% in the hoot split to 51.8245% in the oot split. The train, valid, and test sets maintain consistent target rates near 49.3757%-49.4131%, ensuring reliable model evaluation.

  data_type  count  target_rate
0      hoot   3480     0.428161
1     train   2641     0.494131
2       oot   2576     0.518245
3     valid    881     0.493757
4      test    881     0.493757

## 3. Feature transformation statistics

Optbinning ``BinningProcess`` fit on the ``train`` cohort; WoE columns
derived via ``get_woe_from_bp``. Top entries of the binning table:

                                  name      dtype   status  selected n_bins        iv        js      gini quality_score gini_power
0                 ExternalRiskEstimate  numerical  OPTIMAL      True      6  0.973136  0.113648  0.518808      0.182655   0.518808
1                MSinceOldestTradeOpen  numerical  OPTIMAL      True      6  0.170785  0.021028   0.22314      0.338749    0.77686
2            MSinceMostRecentTradeOpen  numerical  OPTIMAL      True      6  0.050471   0.00624  0.102804      0.021149   0.897196
3                       AverageMInFile  numerical  OPTIMAL      True      6  0.325746   0.03959  0.309368      0.590343   0.690632
4                NumSatisfactoryTrades  numerical  OPTIMAL      True      6  0.080718  0.009961  0.147221      0.089023   0.852779
5          NumTrades60Ever2DerogPubRec  numerical  OPTIMAL      True      4   0.19717  0.024232    0.2038      0.122867     0.7962
6          NumTrades90Ever2DerogPubRec  numerical  OPTIMAL      True      3   0.13669  0.016797  0.150689      0.062318   0.849311
7               PercentTradesNeverDelq  numerical  OPTIMAL      True      6  0.313904  0.038524  0.294309      0.092221   0.705691
8                 MSinceMostRecentDelq  numerical  OPTIMAL      True      5  0.335259  0.040748  0.298855      0.574111   0.701145
9             MaxDelq2PublicRecLast12M  numerical  OPTIMAL      True      4  0.324507  0.039413  0.296656      0.792463   0.703344
10                         MaxDelqEver  numerical  OPTIMAL      True      5  0.232233  0.028712  0.250883      0.016206   0.749117
11                      NumTotalTrades  numerical  OPTIMAL      True      5  0.040309  0.005026  0.109288      0.023917   0.890712
12              NumTradesOpeninLast12M  numerical  OPTIMAL      True      4  0.066104  0.008216   0.14107      0.223403    0.85893
13                PercentInstallTrades  numerical  OPTIMAL      True      6  0.161923  0.019593  0.205908      0.369697   0.794092
14        MSinceMostRecentInqexcl7days  numerical  OPTIMAL      True      6  0.420242  0.050298  0.320313       0.44803   0.679687
15                        NumInqLast6M  numerical  OPTIMAL      True      5    0.1449  0.017814  0.200638      0.233729   0.799362
16               NumInqLast6Mexcl7days  numerical  OPTIMAL      True      5   0.13213  0.016266  0.190939      0.291289   0.809061
17          NetFractionRevolvingBurden  numerical  OPTIMAL      True      6  0.557804  0.066438   0.39176      0.650027    0.60824
18            NetFractionInstallBurden  numerical  OPTIMAL      True      6  0.078145  0.009718  0.153396      0.038576   0.846604
19          NumRevolvingTradesWBalance  numerical  OPTIMAL      True      6  0.083612  0.010359  0.152773       0.10483   0.847227
20            NumInstallTradesWBalance  numerical  OPTIMAL      True      5  0.060135  0.007467  0.132394      0.123245   0.867606
21  NumBank2NatlTradesWHighUtilization  numerical  OPTIMAL      True      5  0.235165  0.028848  0.250296        0.3168   0.749704
22               PercentTradesWBalance  numerical  OPTIMAL      True      6  0.402228  0.048575  0.343265      0.600582   0.656735

## 4. Feature selection statistics

Interpretation: The 'aic_forward' approach is the champion model, utilizing 9 features to achieve a gini_valid of 0.599206 and a gini_test of 0.598113. It outperformed alternative methods, including 'bic_backward' and correlation-based selections, in both validation and testing metrics.

### All branches

         branch_id        recipe  n_features  gini_valid  gini_test  passed
0   br_aic_forward   aic_forward           9    0.599206   0.598113    True
1  br_bic_backward  bic_backward           8    0.596892   0.593603    True
2     br_iv_corr05     iv_corr05          12    0.580996   0.594959    True
3    br_auc_corr05    auc_corr05          12    0.580996   0.594959    True

### Champion features

- `ExternalRiskEstimate_woe`
- `MSinceMostRecentInqexcl7days_woe`
- `AverageMInFile_woe`
- `PercentInstallTrades_woe`
- `NumTotalTrades_woe`
- `NetFractionRevolvingBurden_woe`
- `MSinceMostRecentDelq_woe`
- `NumRevolvingTradesWBalance_woe`
- `NumInqLast6M_woe`

## 5. Model performance

Interpretation: The model performance is robust, with a gini_valid of 0.599206 and a gini_test of 0.598113. The model maintains consistent predictive power across most periods, though it shows a higher performance on the training set (gini 0.650093) compared to the hoot period (gini 0.502860), which recorded the lowest positive rate of 42.8161%.

### Discrimination by cohort

  time_period    aucroc      gini  count  count_positive  positive_rate
0        hoot  0.751430  0.502860   3480            1490       0.428161
1         oot  0.804211  0.608423   2576            1335       0.518245
2        test  0.799057  0.598113    881             435       0.493757
3       train  0.825047  0.650093   2641            1305       0.494131
4       valid  0.799603  0.599206    881             435       0.493757

### Discrimination over time

   time_period    aucroc      gini  count  count_positive  positive_rate
0   2015-01-01  0.790995  0.581990    899             432       0.480534
1   2015-02-01  0.671771  0.343542    812             322       0.396552
2   2015-03-01  0.780892  0.561784    899             318       0.353726
3   2015-04-01  0.744660  0.489319    870             418       0.480460
4   2015-05-01  0.808645  0.617291    899             525       0.583982
5   2015-06-01  0.815566  0.631133    870             426       0.489655
6   2015-07-01  0.826385  0.652771    899             427       0.474972
7   2015-08-01  0.866157  0.732315    895             472       0.527374
8   2015-09-01  0.732759  0.465518    840             325       0.386905
9   2015-10-01  0.785405  0.570810    868             435       0.501152
10  2015-11-01  0.802992  0.605984    840             388       0.461905
11  2015-12-01  0.819509  0.639018    868             512       0.589862

## 6. Model stability

Interpretation: Population Stability Index (PSI) values are generally low, indicating stable model performance across most months, such as 0.001054 in 2015-06-01 and 0.004186 in 2015-08-01. However, notable spikes occurred, particularly in 2015-02-01 with a PSI of 1.079361 and 2015-03-01 with 0.317440, suggesting potential shifts in population characteristics during those periods.

### Score PSI over time

   time_period       psi  count data
0   2015-01-01  0.019300         899
1   2015-02-01  1.079361         812
2   2015-03-01  0.317440         899
3   2015-04-01  0.026159         870
4   2015-05-01  0.100774         899
5   2015-06-01  0.001054         870
6   2015-07-01  0.052855         899
7   2015-08-01  0.004186         895
8   2015-09-01  0.058337         840
9   2015-10-01  0.008340         868
10  2015-11-01  0.040754         840
11  2015-12-01  0.053957         868

## 7. Feature stability

### WoE feature PSI over time

          time                            feature_name       psi  count data
0   2015-01-01                      AverageMInFile_woe  0.006264         899
1   2015-02-01                      AverageMInFile_woe  0.617801         812
2   2015-03-01                      AverageMInFile_woe  0.012261         899
3   2015-04-01                      AverageMInFile_woe  0.008502         870
4   2015-05-01                      AverageMInFile_woe  0.041781         899
5   2015-06-01                      AverageMInFile_woe  0.011916         870
6   2015-07-01                      AverageMInFile_woe  0.001738         899
7   2015-08-01                      AverageMInFile_woe  0.005113         895
8   2015-09-01                      AverageMInFile_woe  0.025685         840
9   2015-10-01                      AverageMInFile_woe  0.008743         868
10  2015-11-01                      AverageMInFile_woe  0.010813         840
11  2015-12-01                      AverageMInFile_woe  0.005255         868
12  2015-01-01                ExternalRiskEstimate_woe  0.001377         899
13  2015-02-01                ExternalRiskEstimate_woe  0.464103         812
14  2015-03-01                ExternalRiskEstimate_woe  0.059126         899
15  2015-04-01                ExternalRiskEstimate_woe  0.004636         870
16  2015-05-01                ExternalRiskEstimate_woe  0.050998         899
17  2015-06-01                ExternalRiskEstimate_woe  0.009057         870
18  2015-07-01                ExternalRiskEstimate_woe  0.020791         899
19  2015-08-01                ExternalRiskEstimate_woe  0.003969         895
20  2015-09-01                ExternalRiskEstimate_woe  0.014831         840
21  2015-10-01                ExternalRiskEstimate_woe  0.015724         868
22  2015-11-01                ExternalRiskEstimate_woe  0.011424         840
23  2015-12-01                ExternalRiskEstimate_woe  0.011837         868
24  2015-01-01                MSinceMostRecentDelq_woe  0.007356         899
25  2015-02-01                MSinceMostRecentDelq_woe  0.304058         812
26  2015-03-01                MSinceMostRecentDelq_woe  0.006239         899
27  2015-04-01                MSinceMostRecentDelq_woe  0.002725         870
28  2015-05-01                MSinceMostRecentDelq_woe  0.004782         899
29  2015-06-01                MSinceMostRecentDelq_woe  0.003129         870
30  2015-07-01                MSinceMostRecentDelq_woe  0.009536         899
31  2015-08-01                MSinceMostRecentDelq_woe  0.009780         895
32  2015-09-01                MSinceMostRecentDelq_woe  0.005299         840
33  2015-10-01                MSinceMostRecentDelq_woe  0.008151         868
34  2015-11-01                MSinceMostRecentDelq_woe  0.002282         840
35  2015-12-01                MSinceMostRecentDelq_woe  0.015728         868
36  2015-01-01        MSinceMostRecentInqexcl7days_woe  0.035564         899
37  2015-02-01        MSinceMostRecentInqexcl7days_woe  1.435269         812
38  2015-03-01        MSinceMostRecentInqexcl7days_woe  4.382147         899
39  2015-04-01        MSinceMostRecentInqexcl7days_woe  0.196115         870
40  2015-05-01        MSinceMostRecentInqexcl7days_woe  1.906819         899
41  2015-06-01        MSinceMostRecentInqexcl7days_woe  0.076676         870
42  2015-07-01        MSinceMostRecentInqexcl7days_woe  0.045300         899
43  2015-08-01        MSinceMostRecentInqexcl7days_woe  0.006091         895
44  2015-09-01        MSinceMostRecentInqexcl7days_woe  0.114466         840
45  2015-10-01        MSinceMostRecentInqexcl7days_woe  0.018772         868
46  2015-11-01        MSinceMostRecentInqexcl7days_woe  0.040520         840
47  2015-12-01        MSinceMostRecentInqexcl7days_woe  0.013636         868
48  2015-01-01           MSinceMostRecentTradeOpen_woe  0.012740         899
49  2015-02-01           MSinceMostRecentTradeOpen_woe  0.250493         812
50  2015-03-01           MSinceMostRecentTradeOpen_woe  0.013082         899
51  2015-04-01           MSinceMostRecentTradeOpen_woe  0.011168         870
52  2015-05-01           MSinceMostRecentTradeOpen_woe  0.003868         899
53  2015-06-01           MSinceMostRecentTradeOpen_woe  0.009986         870
54  2015-07-01           MSinceMostRecentTradeOpen_woe  0.003751         899
55  2015-08-01           MSinceMostRecentTradeOpen_woe  0.015528         895
56  2015-09-01           MSinceMostRecentTradeOpen_woe  0.006583         840
57  2015-10-01           MSinceMostRecentTradeOpen_woe  0.004170         868
58  2015-11-01           MSinceMostRecentTradeOpen_woe  0.017585         840
59  2015-12-01           MSinceMostRecentTradeOpen_woe  0.008910         868
60  2015-01-01               MSinceOldestTradeOpen_woe  0.006466         899
61  2015-02-01               MSinceOldestTradeOpen_woe  0.607798         812
62  2015-03-01               MSinceOldestTradeOpen_woe  0.016901         899
63  2015-04-01               MSinceOldestTradeOpen_woe  0.020376         870
64  2015-05-01               MSinceOldestTradeOpen_woe  0.029658         899
65  2015-06-01               MSinceOldestTradeOpen_woe  0.002836         870
66  2015-07-01               MSinceOldestTradeOpen_woe  0.005976         899
67  2015-08-01               MSinceOldestTradeOpen_woe  0.006098         895
68  2015-09-01               MSinceOldestTradeOpen_woe  0.032143         840
69  2015-10-01               MSinceOldestTradeOpen_woe  0.026940         868
70  2015-11-01               MSinceOldestTradeOpen_woe  0.011988         840
71  2015-12-01               MSinceOldestTradeOpen_woe  0.007016         868
72  2015-01-01            MaxDelq2PublicRecLast12M_woe  0.004630         899
73  2015-02-01            MaxDelq2PublicRecLast12M_woe  1.058743         812
74  2015-03-01            MaxDelq2PublicRecLast12M_woe  0.004001         899
75  2015-04-01            MaxDelq2PublicRecLast12M_woe  0.001819         870
76  2015-05-01            MaxDelq2PublicRecLast12M_woe  0.018210         899
77  2015-06-01            MaxDelq2PublicRecLast12M_woe  0.000750         870
78  2015-07-01            MaxDelq2PublicRecLast12M_woe  0.022378         899
79  2015-08-01            MaxDelq2PublicRecLast12M_woe  0.000077         895
80  2015-09-01            MaxDelq2PublicRecLast12M_woe  0.004593         840
81  2015-10-01            MaxDelq2PublicRecLast12M_woe  0.002323         868
82  2015-11-01            MaxDelq2PublicRecLast12M_woe  0.011973         840
83  2015-12-01            MaxDelq2PublicRecLast12M_woe  0.013663         868
84  2015-01-01                         MaxDelqEver_woe  0.013412         899
85  2015-02-01                         MaxDelqEver_woe  1.057739         812
86  2015-03-01                         MaxDelqEver_woe  0.006677         899
87  2015-04-01                         MaxDelqEver_woe  0.005569         870
88  2015-05-01                         MaxDelqEver_woe  0.028746         899
89  2015-06-01                         MaxDelqEver_woe  0.001526         870
90  2015-07-01                         MaxDelqEver_woe  0.016860         899
91  2015-08-01                         MaxDelqEver_woe  0.003276         895
92  2015-09-01                         MaxDelqEver_woe  0.006195         840
93  2015-10-01                         MaxDelqEver_woe  0.004628         868
94  2015-11-01                         MaxDelqEver_woe  0.010363         840
95  2015-12-01                         MaxDelqEver_woe  0.009667         868
96  2015-01-01            NetFractionInstallBurden_woe  0.001801         899
97  2015-02-01            NetFractionInstallBurden_woe  0.403636         812
98  2015-03-01            NetFractionInstallBurden_woe  0.012351         899
99  2015-04-01            NetFractionInstallBurden_woe  0.002685         870
100 2015-05-01            NetFractionInstallBurden_woe  0.009716         899
101 2015-06-01            NetFractionInstallBurden_woe  0.014589         870
102 2015-07-01            NetFractionInstallBurden_woe  0.011143         899
103 2015-08-01            NetFractionInstallBurden_woe  0.003569         895
104 2015-09-01            NetFractionInstallBurden_woe  0.011910         840
105 2015-10-01            NetFractionInstallBurden_woe  0.003240         868
106 2015-11-01            NetFractionInstallBurden_woe  0.005046         840
107 2015-12-01            NetFractionInstallBurden_woe  0.023387         868
108 2015-01-01          NetFractionRevolvingBurden_woe  0.008771         899
109 2015-02-01          NetFractionRevolvingBurden_woe  0.379290         812
110 2015-03-01          NetFractionRevolvingBurden_woe  0.037802         899
111 2015-04-01          NetFractionRevolvingBurden_woe  0.010787         870
112 2015-05-01          NetFractionRevolvingBurden_woe  0.049365         899
113 2015-06-01          NetFractionRevolvingBurden_woe  0.002055         870
114 2015-07-01          NetFractionRevolvingBurden_woe  0.006289         899
115 2015-08-01          NetFractionRevolvingBurden_woe  0.006317         895
116 2015-09-01          NetFractionRevolvingBurden_woe  0.024580         840
117 2015-10-01          NetFractionRevolvingBurden_woe  0.006390         868
118 2015-11-01          NetFractionRevolvingBurden_woe  0.014881         840
119 2015-12-01          NetFractionRevolvingBurden_woe  0.010521         868
120 2015-01-01  NumBank2NatlTradesWHighUtilization_woe  0.005935         899
121 2015-02-01  NumBank2NatlTradesWHighUtilization_woe  0.199639         812
122 2015-03-01  NumBank2NatlTradesWHighUtilization_woe  0.057295         899
123 2015-04-01  NumBank2NatlTradesWHighUtilization_woe  0.005732         870
124 2015-05-01  NumBank2NatlTradesWHighUtilization_woe  0.006300         899
125 2015-06-01  NumBank2NatlTradesWHighUtilization_woe  0.009002         870
126 2015-07-01  NumBank2NatlTradesWHighUtilization_woe  0.002890         899
127 2015-08-01  NumBank2NatlTradesWHighUtilization_woe  0.002048         895
128 2015-09-01  NumBank2NatlTradesWHighUtilization_woe  0.007238         840
129 2015-10-01  NumBank2NatlTradesWHighUtilization_woe  0.013401         868
130 2015-11-01  NumBank2NatlTradesWHighUtilization_woe  0.014933         840
131 2015-12-01  NumBank2NatlTradesWHighUtilization_woe  0.004843         868
132 2015-01-01                        NumInqLast6M_woe  0.015374         899
133 2015-02-01                        NumInqLast6M_woe  0.300350         812
134 2015-03-01                        NumInqLast6M_woe  0.004897         899
135 2015-04-01                        NumInqLast6M_woe  0.008580         870
136 2015-05-01                        NumInqLast6M_woe  0.006808         899
137 2015-06-01                        NumInqLast6M_woe  0.003935         870
138 2015-07-01                        NumInqLast6M_woe  0.009218         899
139 2015-08-01                        NumInqLast6M_woe  0.009558         895
140 2015-09-01                        NumInqLast6M_woe  0.001960         840
141 2015-10-01                        NumInqLast6M_woe  0.008127         868
142 2015-11-01                        NumInqLast6M_woe  0.006999         840
143 2015-12-01                        NumInqLast6M_woe  0.042671         868
144 2015-01-01               NumInqLast6Mexcl7days_woe  0.012848         899
145 2015-02-01               NumInqLast6Mexcl7days_woe  0.284851         812
146 2015-03-01               NumInqLast6Mexcl7days_woe  0.002592         899
147 2015-04-01               NumInqLast6Mexcl7days_woe  0.009145         870
148 2015-05-01               NumInqLast6Mexcl7days_woe  0.005457         899
149 2015-06-01               NumInqLast6Mexcl7days_woe  0.002441         870
150 2015-07-01               NumInqLast6Mexcl7days_woe  0.008250         899
151 2015-08-01               NumInqLast6Mexcl7days_woe  0.009241         895
152 2015-09-01               NumInqLast6Mexcl7days_woe  0.000927         840
153 2015-10-01               NumInqLast6Mexcl7days_woe  0.012814         868
154 2015-11-01               NumInqLast6Mexcl7days_woe  0.005778         840
155 2015-12-01               NumInqLast6Mexcl7days_woe  0.039487         868
156 2015-01-01            NumInstallTradesWBalance_woe  0.004296         899
157 2015-02-01            NumInstallTradesWBalance_woe  0.960713         812
158 2015-03-01            NumInstallTradesWBalance_woe  0.009407         899
159 2015-04-01            NumInstallTradesWBalance_woe  0.005583         870
160 2015-05-01            NumInstallTradesWBalance_woe  0.005045         899
161 2015-06-01            NumInstallTradesWBalance_woe  0.021461         870
162 2015-07-01            NumInstallTradesWBalance_woe  0.009620         899
163 2015-08-01            NumInstallTradesWBalance_woe  0.004241         895
164 2015-09-01            NumInstallTradesWBalance_woe  0.010968         840
165 2015-10-01            NumInstallTradesWBalance_woe  0.003666         868
166 2015-11-01            NumInstallTradesWBalance_woe  0.016847         840
167 2015-12-01            NumInstallTradesWBalance_woe  0.018302         868
168 2015-01-01          NumRevolvingTradesWBalance_woe  0.006995         899
169 2015-02-01          NumRevolvingTradesWBalance_woe  1.226060         812
170 2015-03-01          NumRevolvingTradesWBalance_woe  0.068108         899
171 2015-04-01          NumRevolvingTradesWBalance_woe  0.028033         870
172 2015-05-01          NumRevolvingTradesWBalance_woe  0.012569         899
173 2015-06-01          NumRevolvingTradesWBalance_woe  0.010624         870
174 2015-07-01          NumRevolvingTradesWBalance_woe  0.022122         899
175 2015-08-01          NumRevolvingTradesWBalance_woe  0.007342         895
176 2015-09-01          NumRevolvingTradesWBalance_woe  0.010788         840
177 2015-10-01          NumRevolvingTradesWBalance_woe  0.006485         868
178 2015-11-01          NumRevolvingTradesWBalance_woe  0.004676         840
179 2015-12-01          NumRevolvingTradesWBalance_woe  0.016497         868
180 2015-01-01               NumSatisfactoryTrades_woe  0.011883         899
181 2015-02-01               NumSatisfactoryTrades_woe  1.201699         812
182 2015-03-01               NumSatisfactoryTrades_woe  0.002968         899
183 2015-04-01               NumSatisfactoryTrades_woe  0.015268         870
184 2015-05-01               NumSatisfactoryTrades_woe  0.039583         899
185 2015-06-01               NumSatisfactoryTrades_woe  0.014542         870
186 2015-07-01               NumSatisfactoryTrades_woe  0.042019         899
187 2015-08-01               NumSatisfactoryTrades_woe  0.004162         895
188 2015-09-01               NumSatisfactoryTrades_woe  0.002100         840
189 2015-10-01               NumSatisfactoryTrades_woe  0.018825         868
190 2015-11-01               NumSatisfactoryTrades_woe  0.019290         840
191 2015-12-01               NumSatisfactoryTrades_woe  0.013723         868
192 2015-01-01                      NumTotalTrades_woe  0.019062         899
193 2015-02-01                      NumTotalTrades_woe  0.887099         812
194 2015-03-01                      NumTotalTrades_woe  0.036568         899
195 2015-04-01                      NumTotalTrades_woe  0.167928         870
196 2015-05-01                      NumTotalTrades_woe  0.062330         899
197 2015-06-01                      NumTotalTrades_woe  0.017953         870
198 2015-07-01                      NumTotalTrades_woe  0.019598         899
199 2015-08-01                      NumTotalTrades_woe  0.009492         895
200 2015-09-01                      NumTotalTrades_woe  0.008062         840
201 2015-10-01                      NumTotalTrades_woe  0.005123         868
202 2015-11-01                      NumTotalTrades_woe  0.023959         840
203 2015-12-01                      NumTotalTrades_woe  0.004393         868
204 2015-01-01         NumTrades60Ever2DerogPubRec_woe  0.006883         899
205 2015-02-01         NumTrades60Ever2DerogPubRec_woe  0.145092         812
206 2015-03-01         NumTrades60Ever2DerogPubRec_woe  0.011829         899
207 2015-04-01         NumTrades60Ever2DerogPubRec_woe  0.003990         870
208 2015-05-01         NumTrades60Ever2DerogPubRec_woe  0.010888         899
209 2015-06-01         NumTrades60Ever2DerogPubRec_woe  0.000040         870
210 2015-07-01         NumTrades60Ever2DerogPubRec_woe  0.005120         899
211 2015-08-01         NumTrades60Ever2DerogPubRec_woe  0.007006         895
212 2015-09-01         NumTrades60Ever2DerogPubRec_woe  0.010208         840
213 2015-10-01         NumTrades60Ever2DerogPubRec_woe  0.000449         868
214 2015-11-01         NumTrades60Ever2DerogPubRec_woe  0.006309         840
215 2015-12-01         NumTrades60Ever2DerogPubRec_woe  0.016924         868
216 2015-01-01         NumTrades90Ever2DerogPubRec_woe  0.008594         899
217 2015-02-01         NumTrades90Ever2DerogPubRec_woe  0.107313         812
218 2015-03-01         NumTrades90Ever2DerogPubRec_woe  0.008829         899
219 2015-04-01         NumTrades90Ever2DerogPubRec_woe  0.001984         870
220 2015-05-01         NumTrades90Ever2DerogPubRec_woe  0.002753         899
221 2015-06-01         NumTrades90Ever2DerogPubRec_woe  0.000599         870
222 2015-07-01         NumTrades90Ever2DerogPubRec_woe  0.001236         899
223 2015-08-01         NumTrades90Ever2DerogPubRec_woe  0.001530         895
224 2015-09-01         NumTrades90Ever2DerogPubRec_woe  0.021485         840
225 2015-10-01         NumTrades90Ever2DerogPubRec_woe  0.000202         868
226 2015-11-01         NumTrades90Ever2DerogPubRec_woe  0.008405         840
227 2015-12-01         NumTrades90Ever2DerogPubRec_woe  0.013174         868
228 2015-01-01              NumTradesOpeninLast12M_woe  0.004572         899
229 2015-02-01              NumTradesOpeninLast12M_woe  0.509379         812
230 2015-03-01              NumTradesOpeninLast12M_woe  0.014055         899
231 2015-04-01              NumTradesOpeninLast12M_woe  0.010655         870
232 2015-05-01              NumTradesOpeninLast12M_woe  0.005899         899
233 2015-06-01              NumTradesOpeninLast12M_woe  0.010686         870
234 2015-07-01              NumTradesOpeninLast12M_woe  0.017167         899
235 2015-08-01              NumTradesOpeninLast12M_woe  0.001787         895
236 2015-09-01              NumTradesOpeninLast12M_woe  0.000866         840
237 2015-10-01              NumTradesOpeninLast12M_woe  0.006127         868
238 2015-11-01              NumTradesOpeninLast12M_woe  0.001720         840
239 2015-12-01              NumTradesOpeninLast12M_woe  0.009201         868
240 2015-01-01                PercentInstallTrades_woe  0.025716         899
241 2015-02-01                PercentInstallTrades_woe  0.703093         812
242 2015-03-01                PercentInstallTrades_woe  0.026718         899
243 2015-04-01                PercentInstallTrades_woe  0.042112         870
244 2015-05-01                PercentInstallTrades_woe  0.053925         899
245 2015-06-01                PercentInstallTrades_woe  0.010961         870
246 2015-07-01                PercentInstallTrades_woe  0.002629         899
247 2015-08-01                PercentInstallTrades_woe  0.008608         895
248 2015-09-01                PercentInstallTrades_woe  0.029570         840
249 2015-10-01                PercentInstallTrades_woe  0.005164         868
250 2015-11-01                PercentInstallTrades_woe  0.090279         840
251 2015-12-01                PercentInstallTrades_woe  0.008827         868
252 2015-01-01              PercentTradesNeverDelq_woe  0.008675         899
253 2015-02-01              PercentTradesNeverDelq_woe  0.871932         812
254 2015-03-01              PercentTradesNeverDelq_woe  0.007613         899
255 2015-04-01              PercentTradesNeverDelq_woe  0.005321         870
256 2015-05-01              PercentTradesNeverDelq_woe  0.048956         899
257 2015-06-01              PercentTradesNeverDelq_woe  0.007492         870
258 2015-07-01              PercentTradesNeverDelq_woe  0.051284         899
259 2015-08-01              PercentTradesNeverDelq_woe  0.002476         895
260 2015-09-01              PercentTradesNeverDelq_woe  0.004396         840
261 2015-10-01              PercentTradesNeverDelq_woe  0.004684         868
262 2015-11-01              PercentTradesNeverDelq_woe  0.013303         840
263 2015-12-01              PercentTradesNeverDelq_woe  0.017245         868
264 2015-01-01               PercentTradesWBalance_woe  0.022284         899
265 2015-02-01               PercentTradesWBalance_woe  0.801028         812
266 2015-03-01               PercentTradesWBalance_woe  0.026827         899
267 2015-04-01               PercentTradesWBalance_woe  0.005777         870
268 2015-05-01               PercentTradesWBalance_woe  0.079869         899
269 2015-06-01               PercentTradesWBalance_woe  0.015746         870
270 2015-07-01               PercentTradesWBalance_woe  0.007823         899
271 2015-08-01               PercentTradesWBalance_woe  0.016590         895
272 2015-09-01               PercentTradesWBalance_woe  0.027200         840
273 2015-10-01               PercentTradesWBalance_woe  0.013002         868
274 2015-11-01               PercentTradesWBalance_woe  0.032770         840
275 2015-12-01               PercentTradesWBalance_woe  0.008214         868

## 8. Final scorecard

Per-feature, per-bin point allocations produced by ``create_scorecard_model``
in ``agent_tools.py``. Each row shows one bin of one feature with its
``Count`` / ``Event rate`` / ``WoE`` / ``IV`` plus the fitted logistic
``Coefficient`` and the final ``Points`` under the PDO scaling (base_score=600, pdo=20, odds=50.0).
To score an application, look up the row matching each of the champion
features for the applicant's value and sum the ``Points`` column.

                        Variable  Bin id               Bin  Count  Count (%)  Non-event  Event  Event rate       WoE        IV            JS  Coefficient     Points
0                 AverageMInFile       0     (-inf, 42.50)    374   0.141613        275     99    0.264706  0.998174  0.129739  1.557596e-02    -0.667830  73.363591
1                 AverageMInFile       1    [42.50, 51.50)    186   0.070428        132     54    0.290323  0.870341  0.049978  6.057209e-03    -0.667830  70.900306
2                 AverageMInFile       2    [51.50, 57.50)    147   0.055661         89     58    0.394558  0.404716  0.008974  1.114094e-03    -0.667830  61.927954
3                 AverageMInFile       3    [57.50, 69.50)    400   0.151458        221    179    0.447500  0.187300  0.005292  6.605407e-04    -0.667830  57.738447
4                 AverageMInFile       4    [69.50, 80.50)    412   0.156002        195    217    0.526699 -0.130375  0.002650  3.310063e-04    -0.667830  51.617014
5                 AverageMInFile       5    [80.50, 97.50)    493   0.186672        210    283    0.574037 -0.321816  0.019204  2.390148e-03    -0.667830  47.928031
6                 AverageMInFile       6   [97.50, 105.50)    149   0.056418         58     91    0.610738 -0.473894  0.012472  1.544602e-03    -0.667830  44.997580
7                 AverageMInFile       7  [105.50, 123.50)    241   0.091253         75    166    0.688797 -0.817977  0.058130  7.070195e-03    -0.667830  38.367269
8                 AverageMInFile       8     [123.50, inf)    239   0.090496         81    158    0.661088 -0.691623  0.041804  5.123840e-03    -0.667830  40.802044
9                 AverageMInFile       9           Special      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00    -0.667830  54.129272
10                AverageMInFile      10           Missing      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00    -0.667830  54.129272
11          ExternalRiskEstimate       0     (-inf, 61.50)    475   0.179856        374    101    0.212632  1.285658  0.260404  3.047904e-02    -0.422275  69.794092
12          ExternalRiskEstimate       1    [61.50, 64.50)    219   0.082923        164     55    0.251142  1.069056  0.086175  1.028660e-02    -0.422275  67.154952
13          ExternalRiskEstimate       2    [64.50, 66.50)    179   0.067777        128     51    0.284916  0.896728  0.050869  6.153852e-03    -0.422275  65.055252
14          ExternalRiskEstimate       3    [66.50, 68.50)    175   0.066263        116     59    0.337143  0.652576  0.027157  3.335687e-03    -0.422275  62.080437
15          ExternalRiskEstimate       4    [68.50, 70.50)    181   0.068535        110     71    0.392265  0.414323  0.011572  1.436207e-03    -0.422275  59.177505
16          ExternalRiskEstimate       5    [70.50, 73.50)    248   0.093904        137    111    0.447581  0.186974  0.003270  4.081171e-04    -0.422275  56.407412
17          ExternalRiskEstimate       6    [73.50, 76.50)    234   0.088603         95    139    0.594017 -0.404074  0.014306  1.776244e-03    -0.422275  49.205921
18          ExternalRiskEstimate       7    [76.50, 80.50)    273   0.103370         80    193    0.706960 -0.904141  0.079576  9.621421e-03    -0.422275  43.112970
19          ExternalRiskEstimate       8    [80.50, 82.50)    142   0.053768         39    103    0.725352 -0.994644  0.049469  5.940730e-03    -0.422275  42.010247
20          ExternalRiskEstimate       9    [82.50, 84.50)    153   0.057933         36    117    0.764706 -1.202132  0.075385  8.893747e-03    -0.422275  39.482159
21          ExternalRiskEstimate      10    [84.50, 86.50)    141   0.053389         27    114    0.808511 -1.463839  0.098292  1.129504e-02    -0.422275  36.293453
22          ExternalRiskEstimate      11      [86.50, inf)    221   0.083680         30    191    0.864253 -1.874553  0.232267  2.541240e-02    -0.422275  31.289193
23          ExternalRiskEstimate      12           Special      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00    -0.422275  54.129272
24          ExternalRiskEstimate      13           Missing      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00    -0.422275  54.129272
25          MSinceMostRecentDelq       0     (-inf, -3.50)   1302   0.492995        498    804    0.617512 -0.502476  0.122271  1.512513e-02    -0.604639  45.362974
26          MSinceMostRecentDelq       1     [-3.50, 2.50)    180   0.068156        140     40    0.222222  1.229286  0.091138  1.072512e-02    -0.604639  75.575635
27          MSinceMostRecentDelq       2     [2.50, 11.50)    402   0.152215        278    124    0.308458  0.783863  0.088627  1.080321e-02    -0.604639  67.804691
28          MSinceMostRecentDelq       3    [11.50, 22.50)    264   0.099962        171     93    0.352273  0.585587  0.033220  4.094186e-03    -0.604639  64.345538
29          MSinceMostRecentDelq       4      [22.50, inf)    493   0.186672        249    244    0.494929 -0.003192  0.000002  2.378048e-07    -0.604639  54.073578
30          MSinceMostRecentDelq       5           Special      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00    -0.604639  54.129272
31          MSinceMostRecentDelq       6           Missing      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00    -0.604639  54.129272
32  MSinceMostRecentInqexcl7days       0     (-inf, -7.50)    188   0.071185         34    154    0.819149 -1.534069  0.141991  1.619057e-02    -0.751279  20.874728
33  MSinceMostRecentInqexcl7days       1    [-7.50, -3.50)    410   0.155244        262    148    0.360976  0.547655  0.045290  5.591533e-03    -0.751279  66.000982
34  MSinceMostRecentInqexcl7days       2     [-3.50, 0.50)   1182   0.447558        730    452    0.382403  0.455885  0.091199  1.130211e-02    -0.751279  64.011655
35  MSinceMostRecentInqexcl7days       3      [0.50, 1.50)    183   0.069292         86     97    0.530055 -0.143841  0.001432  1.788959e-04    -0.751279  51.011187
36  MSinceMostRecentInqexcl7days       4      [1.50, 8.50)    442   0.167361        159    283    0.640271 -0.600020  0.058710  7.230567e-03    -0.751279  41.122438
37  MSinceMostRecentInqexcl7days       5       [8.50, inf)    236   0.089360         65    171    0.724576 -0.990753  0.081620  9.804686e-03    -0.751279  32.652372
38  MSinceMostRecentInqexcl7days       6           Special      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00    -0.751279  54.129272
39  MSinceMostRecentInqexcl7days       7           Missing      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00    -0.751279  54.129272
40    NetFractionRevolvingBurden       0     (-inf, 16.50)   1043   0.394926        343    700    0.671141 -0.736827  0.206062  2.519050e-02    -0.424594  45.102258
41    NetFractionRevolvingBurden       1    [16.50, 23.50)    196   0.074214         80    116    0.591837 -0.395041  0.011460  1.423207e-03    -0.424594  49.289551
42    NetFractionRevolvingBurden       2    [23.50, 35.50)    326   0.123438        160    166    0.509202 -0.060291  0.000449  5.608163e-05    -0.424594  53.390635
43    NetFractionRevolvingBurden       3    [35.50, 51.50)    369   0.139720        223    146    0.395664  0.400088  0.022020  2.734332e-03    -0.424594  59.030832
44    NetFractionRevolvingBurden       4    [51.50, 58.50)    156   0.059069         98     58    0.371795  0.501047  0.014485  1.791883e-03    -0.424594  60.267705
45    NetFractionRevolvingBurden       5    [58.50, 79.50)    325   0.123059        245     80    0.246154  1.095755  0.133770  1.593195e-02    -0.424594  67.553580
46    NetFractionRevolvingBurden       6      [79.50, inf)    226   0.085574        187     39    0.172566  1.544070  0.169979  1.936004e-02    -0.424594  73.045982
47    NetFractionRevolvingBurden       7           Special      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00    -0.424594  54.129272
48    NetFractionRevolvingBurden       8           Missing      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00    -0.424594  54.129272
49                  NumInqLast6M       0      (-inf, 0.50)   1075   0.407043        448    627    0.583256 -0.359630  0.052193  6.489231e-03    -0.295761  51.060237
50                  NumInqLast6M       1      [0.50, 1.50)    652   0.246876        324    328    0.503067 -0.035747  0.000316  3.943603e-05    -0.295761  53.824211
51                  NumInqLast6M       2      [1.50, 2.50)    402   0.152215        228    174    0.432836  0.246813  0.009212  1.148635e-03    -0.295761  56.235542
52                  NumInqLast6M       3      [2.50, 3.50)    220   0.083302        131     89    0.404545  0.363084  0.010840  1.347574e-03    -0.295761  57.227779
53                  NumInqLast6M       4       [3.50, inf)    292   0.110564        205     87    0.297945  0.833625  0.072339  8.789321e-03    -0.295761  61.243309
54                  NumInqLast6M       5           Special      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00    -0.295761  54.129272
55                  NumInqLast6M       6           Missing      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00    -0.295761  54.129272
56    NumRevolvingTradesWBalance       0      (-inf, 0.50)    169   0.063991         91     78    0.461538  0.130674  0.001090  1.361901e-04    -0.582124  56.324138
57    NumRevolvingTradesWBalance       1      [0.50, 1.50)    341   0.129118        147    194    0.568915 -0.300903  0.011624  1.447492e-03    -0.582124  49.075147
58    NumRevolvingTradesWBalance       2      [1.50, 2.50)    483   0.182885        215    268    0.554865 -0.243826  0.010835  1.350982e-03    -0.582124  50.033837
59    NumRevolvingTradesWBalance       3      [2.50, 3.50)    430   0.162817        197    233    0.541860 -0.191312  0.005948  7.423290e-04    -0.582124  50.915895
60    NumRevolvingTradesWBalance       4      [3.50, 4.50)    377   0.142749        185    192    0.509284 -0.060617  0.000525  6.555727e-05    -0.582124  53.111123
61    NumRevolvingTradesWBalance       5      [4.50, 5.50)    265   0.100341        143    122    0.460377  0.135347  0.001834  2.290572e-04    -0.582124  56.402627
62    NumRevolvingTradesWBalance       6      [5.50, 6.50)    176   0.066641        100     76    0.431818  0.250960  0.004169  5.197783e-04    -0.582124  58.344531
63    NumRevolvingTradesWBalance       7       [6.50, inf)    400   0.151458        258    142    0.355000  0.573655  0.048360  5.963456e-03    -0.582124  63.764705
64    NumRevolvingTradesWBalance       8           Special      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00    -0.582124  54.129272
65    NumRevolvingTradesWBalance       9           Missing      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00    -0.582124  54.129272
66                NumTotalTrades       0      (-inf, 8.50)    323   0.122302        185    138    0.427245  0.269625  0.008824  1.099638e-03    -1.437796  65.314947
67                NumTotalTrades       1     [8.50, 17.50)    745   0.282090        416    329    0.441611  0.211150  0.012515  1.561461e-03    -1.437796  62.889067
68                NumTotalTrades       2    [17.50, 21.50)    368   0.139341        184    184    0.500000 -0.023477  0.000077  9.600765e-06    -1.437796  53.155303
69                NumTotalTrades       3    [21.50, 41.50)   1016   0.384703        471    545    0.536417 -0.169405  0.011025  1.376454e-03    -1.437796  47.101343
70                NumTotalTrades       4      [41.50, inf)    189   0.071564         80    109    0.576720 -0.332798  0.007869  9.790988e-04    -1.437796  40.322794
71                NumTotalTrades       5           Special      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00    -1.437796  54.129272
72                NumTotalTrades       6           Missing      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00    -1.437796  54.129272
73          PercentInstallTrades       0     (-inf, 17.50)    495   0.187429        186    309    0.624242 -0.531072  0.051811  6.401371e-03    -0.673950  43.802000
74          PercentInstallTrades       1    [17.50, 23.50)    320   0.121166        142    178    0.556250 -0.249434  0.007511  9.364117e-04    -0.673950  49.278763
75          PercentInstallTrades       2    [23.50, 28.50)    265   0.100341        130    135    0.509434 -0.061217  0.000376  4.699908e-05    -0.673950  52.938833
76          PercentInstallTrades       3    [28.50, 37.50)    502   0.190080        249    253    0.503984 -0.039414  0.000295  3.691087e-05    -0.673950  53.362831
77          PercentInstallTrades       4    [37.50, 40.50)    190   0.071942        100     90    0.473684  0.081883  0.000482  6.021648e-05    -0.673950  55.721587
78          PercentInstallTrades       5    [40.50, 49.50)    334   0.126467        178    156    0.467066  0.108451  0.001485  1.855398e-04    -0.673950  56.238212
79          PercentInstallTrades       6    [49.50, 52.50)    153   0.057933         91     62    0.405229  0.360248  0.007423  9.228428e-04    -0.673950  61.134693
80          PercentInstallTrades       7    [52.50, 65.50)    241   0.091253        150     91    0.377593  0.476299  0.020263  2.509262e-03    -0.673950  63.391426
81          PercentInstallTrades       8      [65.50, inf)    141   0.053389        110     31    0.219858  1.243016  0.072817  8.557921e-03    -0.673950  78.301089
82          PercentInstallTrades       9           Special      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00    -0.673950  54.129272
83          PercentInstallTrades      10           Missing      0   0.000000          0      0    0.000000  0.000000  0.000000  0.000000e+00    -0.673950  54.129272

## 9. Problematic features

- None identified

## 10. Other important content

**Model form:** binary logistic regression on WoE features with PDO scaling
(base_score=600, pdo=20, odds=50.0).

**Reproducibility:** run_id=`20260418T082224-968a7b`; artifacts at `F:\datascience\dsa-agent\predictive-model-agent\scorecard_runs\20260418T082224-968a7b`;
data=`F:\datascience\dsa-agent\predictive-model-agent\data\heloc_dataset_v1.parquet`.

**Champion vs alternates:** champion=`br_aic_forward`; alternates=`br_bic_backward, br_iv_corr05, br_auc_corr05`.

**Deployment notes:** input features map through the frozen ``BinningProcess``;
score column name=`score`.

**Limitations:** sample size is limited to the data window; OOT horizon is
bounded by the ``SplitConfig``; known data defects are listed under
problematic features.
