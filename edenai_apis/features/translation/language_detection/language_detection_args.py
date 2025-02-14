# pylint: disable=locally-disabled, line-too-long
def language_detection_arguments(provider_name: str):
    return {
        "text": """Ogni individuo ha diritto all'istruzione. L'istruzione deve essere gratuita almeno per quanto riguarda le classi elementari e fondamentali. L'istruzione elementare deve essere obbligatoria. L'istruzione tecnica e professionale deve essere messa alla portata di tutti e l'istruzione superiore deve essere egualmente accessibile a tutti sulla base del merito.
L'istruzione deve essere indirizzata al pieno sviluppo della personalità umana ed al rafforzamento del rispetto dei diritti umani e delle libertà fondamentali. Essa deve promuovere la comprensione, la tolleranza, l'amicizia fra tutte le Nazioni, i gruppi razziali e religiosi, e deve favorire l'opera delle Nazioni Unite per il mantenimento della pace.
I genitori hanno diritto di priorità nella scelta del genere di istruzione da impartire ai loro figli.""",
        "settings": {"openai": "gpt-4o", "xai": "grok-2-latest"},
    }
