# Tunisian Sentiment Analysis Model

This model predicts the sentiment of Tunisian text as either Positive or Negative. It works with both Tunisian Arabiz and standard Arabic script.

## What is Tunisian Arabiz? / ما هي العربيزية التونسية؟

Tunisian Arabiz is a way of writing the Tunisian dialect using Latin characters and numbers. For example:
* "3ajbetni" means "I liked it"
* "7aja" means "thing"
* "a3tini 9ahwa" means "give me a coffee"

Try the examples below or enter your own text!
!جرب الأمثلة أو أدخل نصك الخاص

## Usage

[Insert instructions on how to use the model here]

## Examples

1. 3jebni barcha el film hedha
2. ma7abitch el mekla mte3 el restaurant
3. el jaw fi tounes a7la 7aja
4. ennes el kol te3ba w ma3andhomch flous
5. كان جات الدنيا دنيا راني ساهرة في دار حماتي
6. مبابي مانستعرف بيه مدريدي كان مانشوفو مركى هاتريك بمريول الريال

## About the Model

This sentiment analysis model was trained on a combined dataset from TuniziDataset and the Tunisian Dialect Corpus. It uses TF-IDF vectorization for feature extraction and Logistic Regression for classification.

The model accepts Tunisian Arabiz written with Latin and Arabic script.

## Limitations

- Due to dataset limitations, neutral sentiment data was removed to achieve maximum performance.
- The model may not perform well on very colloquial expressions or new slang terms not present in the training data.
- Sentiment can be nuanced and context-dependent, which may not always be captured accurately by this model.

This model is open-source, and contributions of additional datasets are welcome to improve its capabilities.

هذا النموذج مفتوح المصدر، ونرحب بمساهمات مجموعات البيانات الإضافية لتحسين قدراته.
