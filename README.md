# LogisticRegressionToxicity

The idea behind this project is to build a model to determine if a comment on a website is considered "toxic."

The dataset comes from Kaggle.

I decided to keep the method simple and to try a Logistic Regression. I noticed in the official competition that it was posed as a regression problem to determine a a toxicity score of 0-1 and that the leading scores were about 0.96 or so. I turned the problem into a classification problem instead.

My reasoning is that I am again trying to showcase a simple solution to a real world problem. The official competition was done to predict bias and I would rather have a system that just removes toxic comments. I am not a hater of free speech though so I put my toxic level at 0.6. If the toxicity value was at or above 0.6 then it was listed as toxic.

I built a simple Logistic Regression Classifier and achieved an overall accuracy rating of 0.96, again, this model was built and trained in 20 minutes. For real world problems this is fast and relatively elegant as a base to build from here.

I love using Neural Nets but sometimes a simple algorithm can get the job done.

WORK FOR THE FUTURE:
Maybe actually build the regressor.
Try NN with LSTM.
How do SOTA Huggingface perform?
Maybe try NaiveBayes
