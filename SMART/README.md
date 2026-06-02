Code for **DRAMA: Domain Retrieval using Adaptive Module Allocation**

For the academic dataset: <br>
First download the datset from zenodo: https://zenodo.org/records/6606557 <br>
the launch this script: <code>make_all_academic_dataset.py</code>


For the cQA dataset: <br>
First download the datset from zenodo: https://zenodo.org/records/10679181 <br>
the launch this script: <code>make_se_pqa_comm.py</code>

To create the dataset for the unseen domain having downloaded the cQA dataset <br>
run <code>make_se_pqa_extra.py</code> and then <code>merge_se_qrels.py</code>

To recreate the results navigate to the <code>src</code> folder and run the <code>pipeline.sh</code>. 
Change in the model parameters needs can be changed in the pipeline script or in the config files in the <code>conf/</code> folder unless they are overidden in the script files. 
In this case the only way to change the parameters is in the script file.
