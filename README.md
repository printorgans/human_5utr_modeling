requries python 2.7 https://www.python.org/downloads/release/python-2718/

# Human 5′ UTR design and variant effect prediction from a massively parallel translation assay

Predicting the impact of cis-regulatory sequence on gene expression is a foundational challenge for biology. We combine polysome profiling of hundreds of thousands of randomized 5′ UTRs with deep learning to build a predictive model that relates human 5′ UTR sequence to translation. Together with a genetic algorithm, we use the model to engineer new 5′ UTRs that accurately target specified levels of ribosome loading, providing the ability to tune sequences for optimal protein expression. We show that the same approach can be extended to chemically modified RNA, an important feature for applications in mRNA therapeutics and synthetic biology. We test 35,000 truncated human 5′ UTRs and 3,577 naturally-occurring variants and show that the model accurately predicts ribosome loading of these sequences. Finally, we provide evidence of 47 SNVs associated with human diseases that cause a significant change in ribosome loading and thus a plausible molecular basis for disease.

https://www.nature.com/articles/s41587-019-0164-5

<p align="center">
  <img <img src="https://i.ibb.co/vqJjn0D/5-UTR-modeling.png" alt="5-UTR-modeling" border="0"/>
</p>

All data can be found here: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE114002

After cloning the repository, create a new directory named 'data' and place .CSVs from the GEO link above.
