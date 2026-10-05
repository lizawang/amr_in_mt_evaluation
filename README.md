# A Project using AMR (abstract meaning representation) measures as machine translation (MT) quality evaluation metrics
- Sentences are parsed using two AMR parsers: **StructuredBart_large** and **StucturedBART_ensemble** from [IBM github](https://github.com/IBM/transition-amr-parser/tree/master).
-  Include four AMR similarity metrics: [Smatch](https://github.com/snowblink14/smatch), [Smatch++](https://github.com/flipz357/smatchpp#basic-eval), [SemBleu](https://github.com/freesunshine0316/sembleu/tree/master) and [AMRsim](https://github.com/zzshou/AMRSim/blob/main/sentence-transformers/preprocess.py)
-  The comparison is done using the tool, test data and correlation methods from [WMT23 MT Metrics shared Task](https://aclanthology.org/2023.wmt-1.51/)

Details on the implemtation and results please refer to the paper.
