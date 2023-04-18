# Online Companion to  
## ***A Predictive Prescription Framework for Unit Commitment Using Boosting Ensemble Learning Algorithms***

This is the online companion to the publication

> O. Yurdakul, F. Qiu, and A. Meyer "A Predictive Prescription Framework for Unit Commitment Using Boosting Ensemble Learning Algorithms," 
> submitted to IEEE Transactions on Power Systems.

This online companion contains
1. The two [appendices](/appendix.pdf) to the manuscript. In Appendix A, we furnish the notation and the mathematical formulation of the unit commitment problem employed in the publication. In Appendix B, we report the results of two additional case studies conduct to further assess the performance of the proposed framework
2. [Further details on the case studies conducted](#further-details-on-the-case-studies)
3. Source code and the simulation scripts for [(i)](/source_code/ml_files) training the machine learning models and obtaining weights and [(ii)](/source_code/dm_files) deriving unit commitment decisions and evaluating out-of-sample performances. Note that the package 

# Further details on the case studies
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
