# Available Features:
<details><summary>ocr</summary>

| Subfeatures | Providers |
|----------|-------------|
| **identity_parser** | affinda |
| | amazon |
| | base64 |
| | klippa |
| | microsoft |
| | mindee |
| **invoice_parser** | affinda |
| | amazon |
| | base64 |
| | dataleon |
| | google |
| | klippa |
| | microsoft |
| | mindee |
| | rossum |
| | veryfi |
| **receipt_parser** | affinda |
| | amazon |
| | base64 |
| | dataleon |
| | google |
| | klippa |
| | microsoft |
| | mindee |
| | tabscanner |
| | veryfi |
| **resume_parser** | affinda |
| | hireability |
| | klippa |
| **custom_document_parsing_async** | amazon |
| **data_extraction** | amazon |
| | base64 |
| **ocr** | amazon |
| | api4ai |
| | base64 |
| | clarifai |
| | google |
| | microsoft |
| | sentisight |
| **ocr_async** | amazon |
| | google |
| | oneai |
| **ocr_tables_async** | amazon |
| | google |
| | microsoft |
| **bank_check_parsing** | base64 |
| | mindee |
| | veryfi |

</details>
<details><summary>text</summary>

| Subfeatures | Providers |
|----------|-------------|
| **generation** | ai21labs |
| | anthropic |
| | cohere |
| | google |
| | openai |
| **summarize** | alephalpha |
| | cohere |
| | connexun |
| | emvista |
| | huggingface |
| | meaningcloud |
| | microsoft |
| | oneai |
| | openai |
| | writesonic |
| **anonymization** | amazon |
| | emvista |
| | microsoft |
| | oneai |
| | openai |
| **entity_sentiment** | amazon |
| | google |
| **keyword_extraction** | amazon |
| | emvista |
| | ibm |
| | microsoft |
| | nlpcloud |
| | oneai |
| | openai |
| | tenstorrent |
| **named_entity_recognition** | amazon |
| | google |
| | ibm |
| | lettria |
| | microsoft |
| | neuralspace |
| | nlpcloud |
| | oneai |
| | openai |
| | tenstorrent |
| **sentiment_analysis** | amazon |
| | connexun |
| | emvista |
| | google |
| | ibm |
| | lettria |
| | microsoft |
| | nlpcloud |
| | oneai |
| | openai |
| | sapling |
| | tenstorrent |
| **syntax_analysis** | amazon |
| | emvista |
| | google |
| | ibm |
| | lettria |
| **moderation** | clarifai |
| | google |
| | microsoft |
| | openai |
| **custom_classification** | cohere |
| | openai |
| **custom_named_entity_recognition** | cohere |
| | openai |
| **embeddings** | cohere |
| | google |
| | openai |
| **search** | cohere |
| | google |
| | openai |
| **spell_check** | cohere |
| | microsoft |
| | nlpcloud |
| | openai |
| | prowritingaid |
| | sapling |
| **chat** | google |
| | openai |
| | replicate |
| **code_generation** | google |
| | nlpcloud |
| | openai |
| **topic_extraction** | google |
| | ibm |
| | openai |
| | tenstorrent |
| **question_answer** | huggingface |
| | openai |
| | tenstorrent |
| **emotion_detection** | nlpcloud |
| | vernai |
| **prompt_optimization** | openai |
| **ai_detection** | originalityai |
| | sapling |
| | winstonai |
| **plagia_detection** | originalityai |
| | winstonai |

</details>
<details><summary>image</summary>

| Subfeatures | Providers |
|----------|-------------|
| **embeddings** | alephalpha |
| **question_answer** | alephalpha |
| **explicit_content** | amazon |
| | api4ai |
| | clarifai |
| | google |
| | microsoft |
| | picpurify |
| | sentisight |
| **face_compare** | amazon |
| | base64 |
| | facepp |
| **face_detection** | amazon |
| | api4ai |
| | clarifai |
| | google |
| | microsoft |
| | picpurify |
| | skybiometry |
| **face_recognition** | amazon |
| | facepp |
| | microsoft |
| **object_detection** | amazon |
| | api4ai |
| | clarifai |
| | google |
| | microsoft |
| | sentisight |
| **anonymization** | api4ai |
| **logo_detection** | api4ai |
| | clarifai |
| | google |
| | microsoft |
| | smartclick |
| **generation** | deepai |
| | openai |
| | replicate |
| | stabilityai |
| **landmark_detection** | google |
| | microsoft |
| **search** | nyckel |
| | sentisight |

</details>
<details><summary>audio</summary>

| Subfeatures | Providers |
|----------|-------------|
| **speech_to_text_async** | amazon |
| | assembly |
| | deepgram |
| | gladia |
| | google |
| | ibm |
| | microsoft |
| | neuralspace |
| | oneai |
| | openai |
| | revai |
| | speechmatics |
| | symbl |
| | voci |
| | voxist |
| **text_to_speech** | amazon |
| | elevenlabs |
| | google |
| | ibm |
| | lovoai |
| | microsoft |
| **text_to_speech_async** | lovoai |

</details>
<details><summary>translation</summary>

| Subfeatures | Providers |
|----------|-------------|
| **automatic_translation** | amazon |
| | deepl |
| | google |
| | huggingface |
| | ibm |
| | microsoft |
| | modernmt |
| | neuralspace |
| | openai |
| | phedone |
| **language_detection** | amazon |
| | google |
| | ibm |
| | microsoft |
| | modernmt |
| | neuralspace |
| | oneai |
| | openai |
| **document_translation** | deepl |
| | google |

</details>
<details><summary>video</summary>

| Subfeatures | Providers |
|----------|-------------|
| **explicit_content_detection_async** | amazon |
| | google |
| **face_detection_async** | amazon |
| | google |
| **label_detection_async** | amazon |
| | google |
| **person_tracking_async** | amazon |
| | google |
| **text_detection_async** | amazon |
| | google |
| **logo_detection_async** | google |
| **object_tracking_async** | google |

</details>


# Available Providers:
<details><summary>affinda</summary>

| Features | Subfeatures |
|----------|-------------|
| **ocr** | identity_parser |
| | invoice_parser |
| | receipt_parser |
| | resume_parser |

</details>
<details><summary>ai21labs</summary>

| Features | Subfeatures |
|----------|-------------|
| **text** | generation |

</details>
<details><summary>alephalpha</summary>

| Features | Subfeatures |
|----------|-------------|
| **image** | embeddings |
| | question_answer |
| **text** | summarize |

</details>
<details><summary>amazon</summary>

| Features | Subfeatures |
|----------|-------------|
| **audio** | speech_to_text_async |
| | text_to_speech |
| **image** | explicit_content |
| | face_compare |
| | face_detection |
| | face_recognition |
| | object_detection |
| **ocr** | custom_document_parsing_async |
| | data_extraction |
| | identity_parser |
| | invoice_parser |
| | ocr |
| | ocr_async |
| | ocr_tables_async |
| | receipt_parser |
| **text** | anonymization |
| | entity_sentiment |
| | keyword_extraction |
| | named_entity_recognition |
| | sentiment_analysis |
| | syntax_analysis |
| **translation** | automatic_translation |
| | language_detection |
| **video** | explicit_content_detection_async |
| | face_detection_async |
| | label_detection_async |
| | person_tracking_async |
| | text_detection_async |

</details>
<details><summary>anthropic</summary>

| Features | Subfeatures |
|----------|-------------|
| **text** | generation |

</details>
<details><summary>api4ai</summary>

| Features | Subfeatures |
|----------|-------------|
| **image** | anonymization |
| | explicit_content |
| | face_detection |
| | logo_detection |
| | object_detection |
| **ocr** | ocr |

</details>
<details><summary>assembly</summary>

| Features | Subfeatures |
|----------|-------------|
| **audio** | speech_to_text_async |

</details>
<details><summary>base64</summary>

| Features | Subfeatures |
|----------|-------------|
| **image** | face_compare |
| **ocr** | bank_check_parsing |
| | data_extraction |
| | identity_parser |
| | invoice_parser |
| | ocr |
| | receipt_parser |

</details>
<details><summary>clarifai</summary>

| Features | Subfeatures |
|----------|-------------|
| **image** | explicit_content |
| | face_detection |
| | logo_detection |
| | object_detection |
| **ocr** | ocr |
| **text** | moderation |

</details>
<details><summary>cohere</summary>

| Features | Subfeatures |
|----------|-------------|
| **text** | custom_classification |
| | custom_named_entity_recognition |
| | embeddings |
| | generation |
| | search |
| | spell_check |
| | summarize |

</details>
<details><summary>connexun</summary>

| Features | Subfeatures |
|----------|-------------|
| **text** | sentiment_analysis |
| | summarize |

</details>
<details><summary>dataleon</summary>

| Features | Subfeatures |
|----------|-------------|
| **ocr** | invoice_parser |
| | receipt_parser |

</details>
<details><summary>deepai</summary>

| Features | Subfeatures |
|----------|-------------|
| **image** | generation |

</details>
<details><summary>deepgram</summary>

| Features | Subfeatures |
|----------|-------------|
| **audio** | speech_to_text_async |

</details>
<details><summary>deepl</summary>

| Features | Subfeatures |
|----------|-------------|
| **translation** | automatic_translation |
| | document_translation |

</details>
<details><summary>elevenlabs</summary>

| Features | Subfeatures |
|----------|-------------|
| **audio** | text_to_speech |

</details>
<details><summary>emvista</summary>

| Features | Subfeatures |
|----------|-------------|
| **text** | anonymization |
| | keyword_extraction |
| | sentiment_analysis |
| | summarize |
| | syntax_analysis |

</details>
<details><summary>facepp</summary>

| Features | Subfeatures |
|----------|-------------|
| **image** | face_compare |
| | face_recognition |

</details>
<details><summary>gladia</summary>

| Features | Subfeatures |
|----------|-------------|
| **audio** | speech_to_text_async |

</details>
<details><summary>google</summary>

| Features | Subfeatures |
|----------|-------------|
| **audio** | speech_to_text_async |
| | text_to_speech |
| **image** | explicit_content |
| | face_detection |
| | landmark_detection |
| | logo_detection |
| | object_detection |
| **ocr** | invoice_parser |
| | ocr |
| | ocr_async |
| | ocr_tables_async |
| | receipt_parser |
| **text** | chat |
| | code_generation |
| | embeddings |
| | entity_sentiment |
| | generation |
| | moderation |
| | named_entity_recognition |
| | search |
| | sentiment_analysis |
| | syntax_analysis |
| | topic_extraction |
| **translation** | automatic_translation |
| | document_translation |
| | language_detection |
| **video** | explicit_content_detection_async |
| | face_detection_async |
| | label_detection_async |
| | logo_detection_async |
| | object_tracking_async |
| | person_tracking_async |
| | text_detection_async |

</details>
<details><summary>hireability</summary>

| Features | Subfeatures |
|----------|-------------|
| **ocr** | resume_parser |

</details>
<details><summary>huggingface</summary>

| Features | Subfeatures |
|----------|-------------|
| **text** | question_answer |
| | summarize |
| **translation** | automatic_translation |

</details>
<details><summary>ibm</summary>

| Features | Subfeatures |
|----------|-------------|
| **audio** | speech_to_text_async |
| | text_to_speech |
| **text** | keyword_extraction |
| | named_entity_recognition |
| | sentiment_analysis |
| | syntax_analysis |
| | topic_extraction |
| **translation** | automatic_translation |
| | language_detection |

</details>
<details><summary>klippa</summary>

| Features | Subfeatures |
|----------|-------------|
| **ocr** | identity_parser |
| | invoice_parser |
| | receipt_parser |
| | resume_parser |

</details>
<details><summary>lettria</summary>

| Features | Subfeatures |
|----------|-------------|
| **text** | named_entity_recognition |
| | sentiment_analysis |
| | syntax_analysis |

</details>
<details><summary>lovoai</summary>

| Features | Subfeatures |
|----------|-------------|
| **audio** | text_to_speech |
| | text_to_speech_async |

</details>
<details><summary>meaningcloud</summary>

| Features | Subfeatures |
|----------|-------------|
| **text** | summarize |

</details>
<details><summary>microsoft</summary>

| Features | Subfeatures |
|----------|-------------|
| **audio** | speech_to_text_async |
| | text_to_speech |
| **image** | explicit_content |
| | face_detection |
| | face_recognition |
| | landmark_detection |
| | logo_detection |
| | object_detection |
| **ocr** | identity_parser |
| | invoice_parser |
| | ocr |
| | ocr_tables_async |
| | receipt_parser |
| **text** | anonymization |
| | keyword_extraction |
| | moderation |
| | named_entity_recognition |
| | sentiment_analysis |
| | spell_check |
| | summarize |
| **translation** | automatic_translation |
| | language_detection |

</details>
<details><summary>mindee</summary>

| Features | Subfeatures |
|----------|-------------|
| **ocr** | bank_check_parsing |
| | identity_parser |
| | invoice_parser |
| | receipt_parser |

</details>
<details><summary>modernmt</summary>

| Features | Subfeatures |
|----------|-------------|
| **translation** | automatic_translation |
| | language_detection |

</details>
<details><summary>neuralspace</summary>

| Features | Subfeatures |
|----------|-------------|
| **audio** | speech_to_text_async |
| **text** | named_entity_recognition |
| **translation** | automatic_translation |
| | language_detection |

</details>
<details><summary>nlpcloud</summary>

| Features | Subfeatures |
|----------|-------------|
| **text** | code_generation |
| | emotion_detection |
| | keyword_extraction |
| | named_entity_recognition |
| | sentiment_analysis |
| | spell_check |

</details>
<details><summary>nyckel</summary>

| Features | Subfeatures |
|----------|-------------|
| **image** | search |

</details>
<details><summary>oneai</summary>

| Features | Subfeatures |
|----------|-------------|
| **audio** | speech_to_text_async |
| **ocr** | ocr_async |
| **text** | anonymization |
| | keyword_extraction |
| | named_entity_recognition |
| | sentiment_analysis |
| | summarize |
| **translation** | language_detection |

</details>
<details><summary>openai</summary>

| Features | Subfeatures |
|----------|-------------|
| **audio** | speech_to_text_async |
| **image** | generation |
| **text** | anonymization |
| | chat |
| | code_generation |
| | custom_classification |
| | custom_named_entity_recognition |
| | embeddings |
| | generation |
| | keyword_extraction |
| | moderation |
| | named_entity_recognition |
| | prompt_optimization |
| | question_answer |
| | search |
| | sentiment_analysis |
| | spell_check |
| | summarize |
| | topic_extraction |
| **translation** | automatic_translation |
| | language_detection |

</details>
<details><summary>originalityai</summary>

| Features | Subfeatures |
|----------|-------------|
| **text** | ai_detection |
| | plagia_detection |

</details>
<details><summary>phedone</summary>

| Features | Subfeatures |
|----------|-------------|
| **translation** | automatic_translation |

</details>
<details><summary>picpurify</summary>

| Features | Subfeatures |
|----------|-------------|
| **image** | explicit_content |
| | face_detection |

</details>
<details><summary>prowritingaid</summary>

| Features | Subfeatures |
|----------|-------------|
| **text** | spell_check |

</details>
<details><summary>replicate</summary>

| Features | Subfeatures |
|----------|-------------|
| **image** | generation |
| **text** | chat |

</details>
<details><summary>revai</summary>

| Features | Subfeatures |
|----------|-------------|
| **audio** | speech_to_text_async |

</details>
<details><summary>rossum</summary>

| Features | Subfeatures |
|----------|-------------|
| **ocr** | invoice_parser |

</details>
<details><summary>sapling</summary>

| Features | Subfeatures |
|----------|-------------|
| **text** | ai_detection |
| | sentiment_analysis |
| | spell_check |

</details>
<details><summary>sentisight</summary>

| Features | Subfeatures |
|----------|-------------|
| **image** | explicit_content |
| | object_detection |
| | search |
| **ocr** | ocr |

</details>
<details><summary>skybiometry</summary>

| Features | Subfeatures |
|----------|-------------|
| **image** | face_detection |

</details>
<details><summary>smartclick</summary>

| Features | Subfeatures |
|----------|-------------|
| **image** | logo_detection |

</details>
<details><summary>speechmatics</summary>

| Features | Subfeatures |
|----------|-------------|
| **audio** | speech_to_text_async |

</details>
<details><summary>stabilityai</summary>

| Features | Subfeatures |
|----------|-------------|
| **image** | generation |

</details>
<details><summary>symbl</summary>

| Features | Subfeatures |
|----------|-------------|
| **audio** | speech_to_text_async |

</details>
<details><summary>tabscanner</summary>

| Features | Subfeatures |
|----------|-------------|
| **ocr** | receipt_parser |

</details>
<details><summary>tenstorrent</summary>

| Features | Subfeatures |
|----------|-------------|
| **text** | keyword_extraction |
| | named_entity_recognition |
| | question_answer |
| | sentiment_analysis |
| | topic_extraction |

</details>
<details><summary>vernai</summary>

| Features | Subfeatures |
|----------|-------------|
| **text** | emotion_detection |

</details>
<details><summary>veryfi</summary>

| Features | Subfeatures |
|----------|-------------|
| **ocr** | bank_check_parsing |
| | invoice_parser |
| | receipt_parser |

</details>
<details><summary>voci</summary>

| Features | Subfeatures |
|----------|-------------|
| **audio** | speech_to_text_async |

</details>
<details><summary>voxist</summary>

| Features | Subfeatures |
|----------|-------------|
| **audio** | speech_to_text_async |

</details>
<details><summary>winstonai</summary>

| Features | Subfeatures |
|----------|-------------|
| **text** | ai_detection |
| | plagia_detection |

</details>
<details><summary>writesonic</summary>

| Features | Subfeatures |
|----------|-------------|
| **text** | summarize |

</details>
