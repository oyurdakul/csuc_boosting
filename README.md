# Online Companion to  
## ***A Predictive Prescription Framework for Unit Commitment Using Boosting Ensemble Learning Algorithms***

This is the online companion to the publication

> O. Yurdakul, F. Qiu, and A. Meyer "A Predictive Prescription Framework for Unit Commitment Using Boosting Ensemble Learning Algorithms," 
> submitted to IEEE Transactions on Power Systems.

This online companion contains
1. The two [appendices](/appendix.pdf) to the manuscript. In Appendix A, we furnish the notation and the mathematical formulation of the unit commitment problem employed in the publication. In Appendix B, we report the results of two additional case studies conduct to further assess the performance of the proposed framework
2. [Further details on the case studies conducted](#further-details-on-the-case-studies)
3. Source code and the simulation scripts for [(i)](/source_code/ml_files) training the machine learning models and obtaining weights and [(ii)](/source_code/dm_files) deriving unit commitment decisions and evaluating out-of-sample performances. Note that the provided scripts utilize a customized, old version of the [UnitCommitment.jl](https://github.com/ANL-CEEESA/UnitCommitment.jl) package to accommodate the two-stage stochastic setting, which is not provided in this online companion. Nevertheless, in an upcoming release of the UnitCommitment.jl package, two-stage stochastic unit commitment problems will be supported. 

## Further details on the case studies
We next provide additional details on the case studies conducted.
### Weather stations
In the case studies reported in the original manuscript as well as in Appendix B, we use as covariates the temperature measurements harvested from the weather stations located in each zone of the New York Control Area (NYCA). Specifically, the selected weather stations are located in the following zones of the NYCA:

1. Capital
2. Central
3. Dunwoodie
4. Genesee
5. Hudson Valley
6. Long Island
7. Millwood
8. Mohawk Valley
9. North
10. New York City
11. West

### Locational marginal prices
For each method and each observation in the test, we computed the resulting locational marginal prices (LMPs) as a by-product of the optimal (relaxed) UC solution. We next plot the LMPs in hour ending 0800 on September 1, 2019, obtained under each method as part of the Case Study I reported in [Appendix B](/appendix.pdf).

We start with the LMPs obtained under the ideal method:

**LMPs under IUC:**
![ideal](/figs/ideal_time_7.png)

We observe that the LMPs attain relatively similar values, suggesting that the system does not experience congestion at this hour. We next turn to the LMPs under the proposed methods, i.e., $`ab`$-wCSUC, $`gbt`$-wCSUC, $`xgb`$-wCSUC, as well as the $`rf`$-wCSUC method.

**LMPs under $`ab`$-wCSUC:**
![ab_w](/figs/weighted_ab_time_7.png)

**LMPs under $`gbt`$-wCSUC:**
![gbt_w](/figs/weighted_gbt_time_7.png)

**LMPs under $`xgb`$-wCSUC:**
![xgb_w](/figs/weighted_xgb_time_7.png)

**LMPs under $`rf`$-wCSUC:**
![rf_w](/figs/weighted_rf_time_7.png)

Note that the proposed method manages to stave off congestion, delivering moderate LMPs over hour ending 0800.

We next turn to the point-f

