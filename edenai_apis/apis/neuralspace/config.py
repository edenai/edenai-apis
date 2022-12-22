from langcodes import closest_supported_match
from edenai_apis.utils.languages import get_language_name_from_code, load_language_constraints

supported_domains = [
        {
            "code": "en",
            "language": "English",
            "domains": [
                {
                    "domain": "general-v3-australia-default",
                    "accent": "Australia"
                },
                {
                    "domain": "general-v3-australia-latest_long",
                    "accent": "Australia"
                },
                {
                    "domain": "general-v3-canada-default",
                    "accent": "Canada"
                },
                {
                    "domain": "general-v3-ghana-default",
                    "accent": "Ghana"
                },
                {
                    "domain": "general-v3-hong_kong-default",
                    "accent": "Hong Kong"
                },
                {
                    "domain": "general-v3-india-default",
                    "accent": "India"
                },
                {
                    "domain": "general-v3-india-latest_long",
                    "accent": "India"
                },
                {
                    "domain": "general-v3-ireland-default",
                    "accent": "Ireland"
                },
                {
                    "domain": "general-v3-kenya-default",
                    "accent": "Kenya"
                },
                {
                    "domain": "general-v3-new_zealand-default",
                    "accent": "New Zealand"
                },
                {
                    "domain": "general-v3-nigeria-default",
                    "accent": "Nigeria"
                },
                {
                    "domain": "general-v3-pakistan-default",
                    "accent": "Pakistan"
                },
                {
                    "domain": "general-v3-philippines-default",
                    "accent": "Philippines"
                },
                {
                    "domain": "general-v3-singapore-default",
                    "accent": "Singapore"
                },
                {
                    "domain": "general-v3-south_africa-default",
                    "accent": "South Africa"
                },
                {
                    "domain": "general-v3-tanzania-default",
                    "accent": "Tanzania"
                },
                {
                    "domain": "general-v3-united_kingdom-default",
                    "accent": "United Kingdom"
                },
                {
                    "domain": "general-v3-united_kingdom-latest_long",
                    "accent": "United Kingdom"
                },
                {
                    "domain": "general-v3-united_states-default",
                    "accent": "United States"
                },
                {
                    "domain": "general-v3-united_states-latest_long",
                    "accent": "United States"
                }
            ]
        },
        {
            "code": "hi",
            "language": "Hindi",
            "domains": [
                {
                    "domain": "general-v3-india-default",
                    "accent": "India"
                },
                {
                    "domain": "general-v3-india-latest_long",
                    "accent": "India"
                }
            ]
        },
        {
            "code": "sv",
            "language": "Swedish",
            "domains": [
                {
                    "domain": "general-v3-sweden-default",
                    "accent": "Sweden"
                }
            ]
        },
        {
            "code": "ar",
            "language": "Arabic",
            "domains": [
                {
                    "domain": "general-v3-algeria-default",
                    "accent": "Algeria"
                },
                {
                    "domain": "general-v3-algeria-latest_long",
                    "accent": "Algeria"
                },
                {
                    "domain": "general-v3-bahrain-default",
                    "accent": "Bahrain"
                },
                {
                    "domain": "general-v3-bahrain-latest_long",
                    "accent": "Bahrain"
                },
                {
                    "domain": "general-v3-egypt-default",
                    "accent": "Egypt"
                },
                {
                    "domain": "general-v3-egypt-latest_long",
                    "accent": "Egypt"
                },
                {
                    "domain": "general-v3-iraq-default",
                    "accent": "Iraq"
                },
                {
                    "domain": "general-v3-iraq-latest_long",
                    "accent": "Iraq"
                },
                {
                    "domain": "general-v3-israel-default",
                    "accent": "Israel"
                },
                {
                    "domain": "general-v3-israel-latest_long",
                    "accent": "Israel"
                },
                {
                    "domain": "general-v3-jordan-default",
                    "accent": "Jordan"
                },
                {
                    "domain": "general-v3-jordan-latest_long",
                    "accent": "Jordan"
                },
                {
                    "domain": "general-v3-kuwait-default",
                    "accent": "Kuwait"
                },
                {
                    "domain": "general-v3-kuwait-latest_long",
                    "accent": "Kuwait"
                },
                {
                    "domain": "general-v3-lebanon-default",
                    "accent": "Lebanon"
                },
                {
                    "domain": "general-v3-lebanon-latest_long",
                    "accent": "Lebanon"
                },
                {
                    "domain": "general-v3-mauritania-latest_long",
                    "accent": "Mauritania"
                },
                {
                    "domain": "general-v3-morocco-default",
                    "accent": "Morocco"
                },
                {
                    "domain": "general-v3-morocco-latest_long",
                    "accent": "Morocco"
                },
                {
                    "domain": "general-v3-oman-default",
                    "accent": "Oman"
                },
                {
                    "domain": "general-v3-oman-latest_long",
                    "accent": "Oman"
                },
                {
                    "domain": "general-v3-qatar-default",
                    "accent": "Qatar"
                },
                {
                    "domain": "general-v3-qatar-latest_long",
                    "accent": "Qatar"
                },
                {
                    "domain": "general-v3-saudi_arabia-default",
                    "accent": "Saudi Arabia"
                },
                {
                    "domain": "general-v3-saudi_arabia-latest_long",
                    "accent": "Saudi Arabia"
                },
                {
                    "domain": "general-v3-state_of_palestine-default",
                    "accent": "State Of Palestine"
                },
                {
                    "domain": "general-v3-state_of_palestine-latest_long",
                    "accent": "State Of Palestine"
                },
                {
                    "domain": "general-v3-tunisia-default",
                    "accent": "Tunisia"
                },
                {
                    "domain": "general-v3-tunisia-latest_long",
                    "accent": "Tunisia"
                },
                {
                    "domain": "general-v3-united_arab_emirates-default",
                    "accent": "United Arab Emirates"
                },
                {
                    "domain": "general-v3-united_arab_emirates-latest_long",
                    "accent": "United Arab Emirates"
                },
                {
                    "domain": "general-v3-yemen-default",
                    "accent": "Yemen"
                },
                {
                    "domain": "general-v3-yemen-latest_long",
                    "accent": "Yemen"
                }
            ]
        },
        {
            "code": "ru",
            "language": "Russian",
            "domains": [
                {
                    "domain": "general-v3-russia-default",
                    "accent": "Russia"
                },
                {
                    "domain": "general-v3-russia-latest_long",
                    "accent": "Russia"
                }
            ]
        },
        {
            "code": "fr",
            "language": "French",
            "domains": [
                {
                    "domain": "general-v3-belgium-default",
                    "accent": "Belgium"
                },
                {
                    "domain": "general-v3-canada-default",
                    "accent": "Canada"
                },
                {
                    "domain": "general-v3-canada-latest_long",
                    "accent": "Canada"
                },
                {
                    "domain": "general-v3-france-default",
                    "accent": "France"
                },
                {
                    "domain": "general-v3-france-latest_long",
                    "accent": "France"
                },
                {
                    "domain": "general-v3-switzerland-default",
                    "accent": "Switzerland"
                }
            ]
        },
        {
            "code": "uk",
            "language": "Ukrainian",
            "domains": [
                {
                    "domain": "general-v3-ukraine-default",
                    "accent": "Ukraine"
                },
                {
                    "domain": "general-v3-ukraine-latest_long",
                    "accent": "Ukraine"
                }
            ]
        },
        {
            "code": "de",
            "language": "German",
            "domains": [
                {
                    "domain": "general-v3-austria-default",
                    "accent": "Austria"
                },
                {
                    "domain": "general-v3-germany-default",
                    "accent": "Germany"
                },
                {
                    "domain": "general-v3-germany-latest_long",
                    "accent": "Germany"
                },
                {
                    "domain": "general-v3-switzerland-default",
                    "accent": "Switzerland"
                }
            ]
        },
        {
            "code": "el",
            "language": "Greek",
            "domains": [
                {
                    "domain": "general-v3-greece-default",
                    "accent": "Greece"
                }
            ]
        },
        {
            "code": "fa",
            "language": "Persian",
            "domains": [
                {
                    "domain": "general-v3-iran-default",
                    "accent": "Iran"
                }
            ]
        },
        {
            "code": "nl",
            "language": "Dutch",
            "domains": [
                {
                    "domain": "general-v3-belgium-default",
                    "accent": "Belgium"
                },
                {
                    "domain": "general-v3-netherlands-default",
                    "accent": "Netherlands"
                },
                {
                    "domain": "general-v3-netherlands-latest_long",
                    "accent": "Netherlands"
                }
            ]
        },
        {
            "code": "pt",
            "language": "Portuguese",
            "domains": [
                {
                    "domain": "general-v3-brazil-default",
                    "accent": "Brazil"
                },
                {
                    "domain": "general-v3-brazil-latest_long",
                    "accent": "Brazil"
                },
                {
                    "domain": "general-v3-portugal-default",
                    "accent": "Portugal"
                },
                {
                    "domain": "general-v3-portugal-latest_long",
                    "accent": "Portugal"
                }
            ]
        },
        {
            "code": "es",
            "language": "Spanish",
            "domains": [
                {
                    "domain": "general-v3-argentina-default",
                    "accent": "Argentina"
                },
                {
                    "domain": "general-v3-bolivia-default",
                    "accent": "Bolivia"
                },
                {
                    "domain": "general-v3-chile-default",
                    "accent": "Chile"
                },
                {
                    "domain": "general-v3-colombia-default",
                    "accent": "Colombia"
                },
                {
                    "domain": "general-v3-costa_rica-default",
                    "accent": "Costa Rica"
                },
                {
                    "domain": "general-v3-dominican_republic-default",
                    "accent": "Dominican Republic"
                },
                {
                    "domain": "general-v3-ecuador-default",
                    "accent": "Ecuador"
                },
                {
                    "domain": "general-v3-el_salvador-default",
                    "accent": "El Salvador"
                },
                {
                    "domain": "general-v3-guatemala-default",
                    "accent": "Guatemala"
                },
                {
                    "domain": "general-v3-honduras-default",
                    "accent": "Honduras"
                },
                {
                    "domain": "general-v3-mexico-default",
                    "accent": "Mexico"
                },
                {
                    "domain": "general-v3-nicaragua-default",
                    "accent": "Nicaragua"
                },
                {
                    "domain": "general-v3-panama-default",
                    "accent": "Panama"
                },
                {
                    "domain": "general-v3-paraguay-default",
                    "accent": "Paraguay"
                },
                {
                    "domain": "general-v3-peru-default",
                    "accent": "Peru"
                },
                {
                    "domain": "general-v3-puerto_rico-default",
                    "accent": "Puerto Rico"
                },
                {
                    "domain": "general-v3-spain-default",
                    "accent": "Spain"
                },
                {
                    "domain": "general-v3-spain-latest_long",
                    "accent": "Spain"
                },
                {
                    "domain": "general-v3-united_states-default",
                    "accent": "United States"
                },
                {
                    "domain": "general-v3-united_states-latest_long",
                    "accent": "United States"
                },
                {
                    "domain": "general-v3-uruguay-default",
                    "accent": "Uruguay"
                },
                {
                    "domain": "general-v3-venezuela-default",
                    "accent": "Venezuela"
                }
            ]
        },
        {
            "code": "ca",
            "language": "Catalan",
            "domains": [
                {
                    "domain": "general-v3-spain-default",
                    "accent": "Spain"
                }
            ]
        },
        {
            "code": "cs",
            "language": "Czech",
            "domains": [
                {
                    "domain": "general-v3-czech_republic-default",
                    "accent": "Czech Republic"
                }
            ]
        },
        {
            "code": "ja",
            "language": "Japanese",
            "domains": [
                {
                    "domain": "general-v3-japan-default",
                    "accent": "Japan"
                },
                {
                    "domain": "general-v3-japan-latest_long",
                    "accent": "Japan"
                }
            ]
        },
        {
            "code": "kk",
            "language": "Kazakh",
            "domains": [
                {
                    "domain": "general-v3-kazakhstan-default",
                    "accent": "Kazakhstan"
                }
            ]
        },
        {
            "code": "tr",
            "language": "Turkish",
            "domains": [
                {
                    "domain": "general-v3-turkey-default",
                    "accent": "Turkey"
                },
                {
                    "domain": "general-v3-turkey-latest_long",
                    "accent": "Turkey"
                }
            ]
        },
        {
            "code": "it",
            "language": "Italian",
            "domains": [
                {
                    "domain": "general-v3-italy-default",
                    "accent": "Italy"
                },
                {
                    "domain": "general-v3-italy-latest_long",
                    "accent": "Italy"
                },
                {
                    "domain": "general-v3-switzerland-default",
                    "accent": "Switzerland"
                }
            ]
        },
        {
            "code": "af",
            "language": "Afrikaans",
            "domains": [
                {
                    "domain": "general-v3-south_africa-default",
                    "accent": "South Africa"
                }
            ]
        },
        {
            "code": "am",
            "domains": [
                {
                    "domain": "general-v3-ethiopia-default",
                    "accent": "Ethiopia"
                }
            ]
        },
        {
            "code": "az",
            "language": "Azerbaijani",
            "domains": [
                {
                    "domain": "general-v3-azerbaijan-default",
                    "accent": "Azerbaijan"
                }
            ]
        },
        {
            "code": "bg",
            "language": "Bulgarian",
            "domains": [
                {
                    "domain": "general-v3-bulgaria-default",
                    "accent": "Bulgaria"
                }
            ]
        },
        {
            "code": "bn",
            "language": "Bengali",
            "domains": [
                {
                    "domain": "general-v3-bangladesh-default",
                    "accent": "Bangladesh"
                },
                {
                    "domain": "general-v3-india-default",
                    "accent": "India"
                }
            ]
        },
        {
            "code": "bs",
            "language": "Bosnian",
            "domains": [
                {
                    "domain": "general-v3-bosnia_and_herzegovina-default",
                    "accent": "Bosnia And Herzegovina"
                }
            ]
        },
        {
            "code": "da",
            "language": "Danish",
            "domains": [
                {
                    "domain": "general-v3-denmark-default",
                    "accent": "Denmark"
                },
                {
                    "domain": "general-v3-denmark-latest_long",
                    "accent": "Denmark"
                }
            ]
        },
        {
            "code": "et",
            "language": "Estonian",
            "domains": [
                {
                    "domain": "general-v3-estonia-default",
                    "accent": "Estonia"
                }
            ]
        },
        {
            "code": "eu",
            "language": "Basque",
            "domains": [
                {
                    "domain": "general-v3-spain-default",
                    "accent": "Spain"
                }
            ]
        },
        {
            "code": "fi",
            "language": "Finnish",
            "domains": [
                {
                    "domain": "general-v3-finland-default",
                    "accent": "Finland"
                },
                {
                    "domain": "general-v3-finland-latest_long",
                    "accent": "Finland"
                }
            ]
        },
        {
            "code": "gl",
            "language": "Galician",
            "domains": [
                {
                    "domain": "general-v3-spain-default",
                    "accent": "Spain"
                }
            ]
        },
        {
            "code": "gu",
            "language": "Gujarati",
            "domains": [
                {
                    "domain": "general-v3-india-default",
                    "accent": "India"
                }
            ]
        },
        {
            "code": "hr",
            "language": "Croatian",
            "domains": [
                {
                    "domain": "general-v3-croatia-default",
                    "accent": "Croatia"
                }
            ]
        },
        {
            "code": "hu",
            "language": "Hungarian",
            "domains": [
                {
                    "domain": "general-v3-hungary-default",
                    "accent": "Hungary"
                }
            ]
        },
        {
            "code": "hy",
            "language": "Armenian",
            "domains": [
                {
                    "domain": "general-v3-armenia-default",
                    "accent": "Armenia"
                }
            ]
        },
        {
            "code": "id",
            "language": "Indonesian",
            "domains": [
                {
                    "domain": "general-v3-indonesia-default",
                    "accent": "Indonesia"
                },
                {
                    "domain": "general-v3-indonesia-latest_long",
                    "accent": "Indonesia"
                }
            ]
        },
        {
            "code": "is",
            "language": "Icelandic",
            "domains": [
                {
                    "domain": "general-v3-iceland-default",
                    "accent": "Iceland"
                }
            ]
        },
        {
            "code": "jv",
            "language": "Javanese",
            "domains": [
                {
                    "domain": "general-v3-indonesia-default",
                    "accent": "Indonesia"
                }
            ]
        },
        {
            "code": "ka",
            "language": "Georgian",
            "domains": [
                {
                    "domain": "general-v3-georgia-default",
                    "accent": "Georgia"
                }
            ]
        },
        {
            "code": "km",
            "domains": [
                {
                    "domain": "general-v3-cambodia-default",
                    "accent": "Cambodia"
                }
            ]
        },
        {
            "code": "kn",
            "language": "Kannada",
            "domains": [
                {
                    "domain": "general-v3-india-default",
                    "accent": "India"
                }
            ]
        },
        {
            "code": "ko",
            "language": "Korean",
            "domains": [
                {
                    "domain": "general-v3-south_korea-default",
                    "accent": "South Korea"
                },
                {
                    "domain": "general-v3-south_korea-latest_long",
                    "accent": "South Korea"
                }
            ]
        },
        {
            "code": "lo",
            "domains": [
                {
                    "domain": "general-v3-laos-default",
                    "accent": "Laos"
                }
            ]
        },
        {
            "code": "lt",
            "language": "Lithuanian",
            "domains": [
                {
                    "domain": "general-v3-lithuania-default",
                    "accent": "Lithuania"
                }
            ]
        },
        {
            "code": "lv",
            "language": "Latvian",
            "domains": [
                {
                    "domain": "general-v3-latvia-default",
                    "accent": "Latvia"
                }
            ]
        },
        {
            "code": "mk",
            "language": "Macedonian",
            "domains": [
                {
                    "domain": "general-v3-north_macedonia-default",
                    "accent": "North Macedonia"
                },
                {
                    "domain": "general-v3-north_macedonia-latest_long",
                    "accent": "North Macedonia"
                }
            ]
        },
        {
            "code": "ml",
            "language": "Malayalam",
            "domains": [
                {
                    "domain": "general-v3-india-default",
                    "accent": "India"
                }
            ]
        },
        {
            "code": "mn",
            "domains": [
                {
                    "domain": "general-v3-mongolia-default",
                    "accent": "Mongolia"
                }
            ]
        },
        {
            "code": "mr",
            "language": "Marathi",
            "domains": [
                {
                    "domain": "general-v3-india-default",
                    "accent": "India"
                }
            ]
        },
        {
            "code": "ms",
            "language": "Malay",
            "domains": [
                {
                    "domain": "general-v3-malaysia-default",
                    "accent": "Malaysia"
                }
            ]
        },
        {
            "code": "my",
            "language": "Burmese",
            "domains": [
                {
                    "domain": "general-v3-myanmar-default",
                    "accent": "Myanmar"
                }
            ]
        },
        {
            "code": "ne",
            "language": "Nepali",
            "domains": [
                {
                    "domain": "general-v3-nepal-default",
                    "accent": "Nepal"
                }
            ]
        },
        {
            "code": "no",
            "domains": [
                {
                    "domain": "general-v3-norway-default",
                    "accent": "Norway"
                },
                {
                    "domain": "general-v3-norway-latest_long",
                    "accent": "Norway"
                }
            ]
        },
        {
            "code": "pa",
            "language": "Punjabi",
            "domains": [
                {
                    "domain": "general-v3-gurmukhi_india-default",
                    "accent": "Gurmukhi India"
                }
            ]
        },
        {
            "code": "pl",
            "language": "Polish",
            "domains": [
                {
                    "domain": "general-v3-poland-default",
                    "accent": "Poland"
                },
                {
                    "domain": "general-v3-poland-latest_long",
                    "accent": "Poland"
                }
            ]
        },
        {
            "code": "ro",
            "language": "Romanian",
            "domains": [
                {
                    "domain": "general-v3-romania-default",
                    "accent": "Romania"
                },
                {
                    "domain": "general-v3-romania-latest_long",
                    "accent": "Romania"
                }
            ]
        },
        {
            "code": "si",
            "domains": [
                {
                    "domain": "general-v3-sri_lanka-default",
                    "accent": "Sri Lanka"
                }
            ]
        },
        {
            "code": "sk",
            "language": "Slovak",
            "domains": [
                {
                    "domain": "general-v3-slovakia-default",
                    "accent": "Slovakia"
                }
            ]
        },
        {
            "code": "sl",
            "language": "Slovenian",
            "domains": [
                {
                    "domain": "general-v3-slovenia-default",
                    "accent": "Slovenia"
                }
            ]
        },
        {
            "code": "sq",
            "language": "Albanian",
            "domains": [
                {
                    "domain": "general-v3-albania-default",
                    "accent": "Albania"
                }
            ]
        },
        {
            "code": "sr",
            "language": "Serbian",
            "domains": [
                {
                    "domain": "general-v3-serbia-default",
                    "accent": "Serbia"
                }
            ]
        },
        {
            "code": "su",
            "language": "Sundanese",
            "domains": [
                {
                    "domain": "general-v3-indonesia-default",
                    "accent": "Indonesia"
                }
            ]
        },
        {
            "code": "sw",
            "language": "Swahili",
            "domains": [
                {
                    "domain": "general-v3-kenya-default",
                    "accent": "Kenya"
                },
                {
                    "domain": "general-v3-tanzania-default",
                    "accent": "Tanzania"
                }
            ]
        },
        {
            "code": "ta",
            "language": "Tamil",
            "domains": [
                {
                    "domain": "general-v3-india-default",
                    "accent": "India"
                },
                {
                    "domain": "general-v3-malaysia-default",
                    "accent": "Malaysia"
                },
                {
                    "domain": "general-v3-singapore-default",
                    "accent": "Singapore"
                },
                {
                    "domain": "general-v3-sri_lanka-default",
                    "accent": "Sri Lanka"
                }
            ]
        },
        {
            "code": "te",
            "language": "Telugu",
            "domains": [
                {
                    "domain": "general-v3-india-default",
                    "accent": "India"
                }
            ]
        },
        {
            "code": "th",
            "domains": [
                {
                    "domain": "general-v3-thailand-default",
                    "accent": "Thailand"
                },
                {
                    "domain": "general-v3-thailand-latest_long",
                    "accent": "Thailand"
                }
            ]
        },
        {
            "code": "ur",
            "language": "Urdu",
            "domains": [
                {
                    "domain": "general-v3-india-default",
                    "accent": "India"
                },
                {
                    "domain": "general-v3-pakistan-default",
                    "accent": "Pakistan"
                }
            ]
        },
        {
            "code": "uz",
            "language": "Uzbek",
            "domains": [
                {
                    "domain": "general-v3-uzbekistan-default",
                    "accent": "Uzbekistan"
                }
            ]
        },
        {
            "code": "vi",
            "language": "Vietnamese",
            "domains": [
                {
                    "domain": "general-v3-vietnam-default",
                    "accent": "Vietnam"
                },
                {
                    "domain": "general-v3-vietnam-latest_long",
                    "accent": "Vietnam"
                }
            ]
        },
        {
            "code": "zu",
            "domains": [
                {
                    "domain": "general-v3-south_africa-default",
                    "accent": "South Africa"
                }
            ]
        }
    ]


def get_domain_language_from_code(lang_code):
    if not lang_code:
        return None
    list_languages = load_language_constraints("neuralspace", "audio", "speech_to_text_async")
    list_languages_four_digits = [lang for lang in list_languages if len(lang) > 2]
    language = lang_code
    call_iteraction = 100
    if len(lang_code) == 2:
        while call_iteraction > 0:
            try:
                lang_code = closest_supported_match(lang_code, list_languages_four_digits)
                break
            except RuntimeError:
                call_iteraction -=1
    if lang_code:
        code = lang_code[:2]
        language_name = get_language_name_from_code(lang_code)
        for domain in supported_domains:
            if domain.get('code') == code:
                domains = domain.get('domains')
                try:
                    domain_language = next(filter(lambda dom: f"({dom.get('accent')}" in language_name, domains))
                    if domain_language: 
                        return {
                            "language": domain.get('code'),
                            "domain" : domain_language.get('domain')
                        }
                except StopIteration:
                    pass
    else:
        for domain in supported_domains:
            if domain.get('code') == language:
                domain_language = domain.get('domains')[0]
                return {
                        "language": domain.get('code'),
                        "domain" : domain_language.get('domain')
                    }