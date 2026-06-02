# SMART 

Code for **SMART: Scalable Multi-domain Adaptive ReTrieval**

For the academic dataset: <br>
First download the datset from zenodo: https://zenodo.org/records/6606557 <br>
the launch this script: <code>make_all_academic_dataset.py</code>


For the cQA dataset: <br>
First download the datset from zenodo: https://zenodo.org/records/10679181 <br>
the launch this script: <code>make_se_pqa_comm.py</code>

To create the dataset for the unseen domain having downloaded the cQA dataset <br>
run <code>make_se_pqa_extra.py</code> and then <code>merge_se_qrels.py</code>

To recreate the results on Academic Search and SE-PQA benchmarks, navigate to the <code>src</code> folder and run the <code>pipeline.sh</code>. 
Change in the model parameters needs can be changed in the pipeline script or in the config files in the <code>conf/</code> folder unless they are overidden in the script files. 
In this case the only way to change the parameters is in the script file.

To recreate the results on BEIR and on the baseline, navigate to the <code>evaluate_baseline</code> and run the <code>.sh</code> files in the folder. 
