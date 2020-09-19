# fairness_research

Research on fairness in scientific research.

Checking if there is significant group disparity for recommendations between different 'sensitive' groups 

Link: https://www.overleaf.com/project/5d4dd46e95d6ab6eafe94faa


bag_of_words.py takes two file names as command line arguments and generates the unigram (or bigram) bag of words vector for them.

Sample Arguments:

For main.py 
--path "../../workingAmir/data_info/loaded_pickles_nips19/" --save_model "lda"--epochs "20"]

For evaluate.py
--data_path "../../workingAmir/data_info/loaded_pickles_nips19/"--saved_models "lda_old/"--model_name "LDA-Match_LR-flag_attn-True-epochs-40-batch_size-64-KL-False"]

For rank_app.py
--path "../../workingAmir/data_info/loaded_pickles_nips19/" --save_model "rank_net" --epochs "20"

Folder_reviewer_expertise has:
 models.py, other_models.py (old models) ,utilities
 plots.py

