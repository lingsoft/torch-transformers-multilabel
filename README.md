# torch-transformers-multilabel

This codebase was built with PyTorch 1.11, Transformers 4.18.0 and Hugging Face datasets. 

To train a model, run

    python3 train.py --train [TRAIN_LANGUAGE] --dev [TRAIN_LANGUAGE] --test [TEST_LANGUAGE]
    
Use several languages in training or testing by connecting them with a hyphen. E.g., en-fi-sv.
  
With the slurm script, you can launch multiple instances:

    sbatch slurm_train_arg.sh [TRAIN_LANGUAGE] [TEST_LANGUAGE] [LRs] [EPOCHSs] [INSTANCEs]
    
To predict labels for a text file (several texts when separated by linebreak):
    
    python3 predict.py --text text.txt --load_model model.pt

List of registers and their abbreviations (used as labels):

| General Register                   | Label        |
|------------------------------------|--------------|
| How-to/Instructions                | HI           |
| Interactive discussion             | ID           |
| Informational description          | IN           |
| Informational persuasion           | IP           |
| Lyrical                            | LY           |
| Narrative                          | NA           |
| Opinion                            | OP           |
| Spoken                             | SP           |

| Sub-register                       | Label        | General |
|------------------------------------|--------------|---------|
| Advice                             | av           | OP      |
| Description with intent to sell    | ds           | IP      |
| Description of a thing or a person | dtp          | IN      |
| News & opinion blog or editorial   | ed           | NA      |
| Encyclopedia article               | en           | IN      |
| FAQ                                | fi           | IN      |
| Interview                          | it           | SP      |
| Legal terms and conditions         | lt           | IN      |
| Narrative blog                     | nb           | NA      |
| News report                        | ne           | NA      |
| Opinion blog                       | ob           | OP      |
| Research article                   | ra           | IN      |
| Recipe                             | re           | HI      |
| Religious blog / sermon            | rs           | OP      |
| Review                             | rv           | OP      |
| Sports report                      | sr           | NA      |
