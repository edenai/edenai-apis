# Available Features:
<details><summary>ocr</summary>

| Subfeatures | Providers |
|----------|-------------|
| **invoice_parser** | affinda |
| | base64 |
| | dataleon |
| | microsoft |
| | mindee |
| **resume_parser** | affinda |
| | hireability |
| **identity_parser** | amazon |
| | base64 |
| | microsoft |
| | mindee |
| **ocr** | amazon |
| | api4ai |
| | base64 |
| | clarifai |
| | google |
| | microsoft |
| | sentisight |
| **ocr_tables_async** | amazon |
| | google |
| | microsoft |
| **receipt_parser** | base64 |
| | dataleon |
| | microsoft |
| | mindee |
| | tabscanner |

</details>
<details><summary>audio</summary>

| Subfeatures | Providers |
|----------|-------------|
| **speech_to_text_async** | amazon |
| | assembly |
| | deepgram |
| | google |
| | ibm |
| | microsoft |
| | neuralspace |
| | oneai |
| | revai |
| | symbl |
| | voci |
| | voxist |
| **text_to_speech** | amazon |
| | google |
| | ibm |
| | microsoft |

</details>
<details><summary>image</summary>

| Subfeatures | Providers |
|----------|-------------|
| **explicit_content** | amazon |
| | api4ai |
| | clarifai |
| | google |
| | microsoft |
| | picpurify |
| | sentisight |
| **face_detection** | amazon |
| | api4ai |
| | clarifai |
| | google |
| | microsoft |
| | picpurify |
| **object_detection** | amazon |
| | api4ai |
| | clarifai |
| | google |
| | microsoft |
| | sentisight |
| **anonymization** | api4ai |
| **logo_detection** | api4ai |
| | google |
| | microsoft |
| | smartclick |
| **landmark_detection** | google |
| | microsoft |
| **search** | sentisight |

</details>
<details><summary>text</summary>

| Subfeatures | Providers |
|----------|-------------|
| **keyword_extraction** | amazon |
| | emvista |
| | ibm |
| | microsoft |
| | oneai |
| | openai |
| **named_entity_recognition** | amazon |
| | google |
| | ibm |
| | lettria |
| | microsoft |
| | neuralspace |
| | oneai |
| **sentiment_analysis** | amazon |
| | connexun |
| | emvista |
| | google |
| | ibm |
| | lettria |
| | microsoft |
| | oneai |
| | openai |
| **syntax_analysis** | amazon |
| | emvista |
| | google |
| | ibm |
| | lettria |
| **generation** | cohere |
| | openai |
| **summarize** | connexun |
| | emvista |
| | huggingface |
| | meaningcloud |
| | microsoft |
| | oneai |
| | openai |
| **anonymization** | emvista |
| | oneai |
| | openai |
| **topic_extraction** | google |
| | ibm |
| | openai |
| **question_answer** | huggingface |
| | openai |
| **search** | openai |

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
| **ocr** | invoice_parser |
| | resume_parser |

</details>
<details><summary>amazon</summary>

| Features | Subfeatures |
|----------|-------------|
| **audio** | speech_to_text_async |
| | text_to_speech |
| **image** | explicit_content |
| | face_detection |
| | object_detection |
| **ocr** | identity_parser |
| | ocr |
| | ocr_tables_async |
| **text** | keyword_extraction |
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
| **ocr** | identity_parser |
| | invoice_parser |
| | ocr |
| | receipt_parser |

</details>
<details><summary>clarifai</summary>

| Features | Subfeatures |
|----------|-------------|
| **image** | explicit_content |
| | face_detection |
| | object_detection |
| **ocr** | ocr |

</details>
<details><summary>cohere</summary>

| Features | Subfeatures |
|----------|-------------|
| **text** | generation |

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
<details><summary>deepgram</summary>

| Features | Subfeatures |
|----------|-------------|
| **audio** | speech_to_text_async |

</details>
<details><summary>deepl</summary>

| Features | Subfeatures |
|----------|-------------|
| **translation** | automatic_translation |

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
| **ocr** | ocr |
| | ocr_tables_async |
| **text** | named_entity_recognition |
| | sentiment_analysis |
| | syntax_analysis |
| | topic_extraction |
| **translation** | automatic_translation |
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
<details><summary>lettria</summary>

| Features | Subfeatures |
|----------|-------------|
| **text** | named_entity_recognition |
| | sentiment_analysis |
| | syntax_analysis |

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
| | landmark_detection |
| | logo_detection |
| | object_detection |
| **ocr** | identity_parser |
| | invoice_parser |
| | ocr |
| | ocr_tables_async |
| | receipt_parser |
| **text** | keyword_extraction |
| | named_entity_recognition |
| | sentiment_analysis |
| | summarize |
| **translation** | automatic_translation |
| | language_detection |

</details>
<details><summary>mindee</summary>

| Features | Subfeatures |
|----------|-------------|
| **ocr** | identity_parser |
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
<details><summary>oneai</summary>

| Features | Subfeatures |
|----------|-------------|
| **audio** | speech_to_text_async |
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
| **text** | anonymization |
| | generation |
| | keyword_extraction |
| | question_answer |
| | search |
| | sentiment_analysis |
| | summarize |
| | topic_extraction |
| **translation** | automatic_translation |
| | language_detection |

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
<details><summary>revai</summary>

| Features | Subfeatures |
|----------|-------------|
| **audio** | speech_to_text_async |

</details>
<details><summary>sentisight</summary>

| Features | Subfeatures |
|----------|-------------|
| **image** | explicit_content |
| | object_detection |
| | search |
| **ocr** | ocr |

</details>
<details><summary>smartclick</summary>

| Features | Subfeatures |
|----------|-------------|
| **image** | logo_detection |

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
