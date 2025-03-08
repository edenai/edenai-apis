class Entities:

    @classmethod
    def get_entity(cls, value):
        entitiy_cats = [
            attr
            for attr in dir(cls)
            if not callable(getattr(cls, attr)) and not attr.startswith("_")
        ]
        for ent in entitiy_cats:
            if value in getattr(cls, ent):
                return ent
        return "OTHER"

    UNKNOWN = ["UNKNOWN"]
    PERSON = ["PERSON"]
    LOCATION = ["LOCATION"]
    ORGANIZATION = ["ORGANIZATION"]
    EVENT = ["EVENT"]
    WORK_OF_ART = ["WORK_OF_ART"]
    CONSUMER_GOOD = ["CONSUMER_GOOD", "COMMERCIAL_ITEM"]
    PHONE_NUMBER = ["PHONE_NUMBER"]
    ADDRESS = ["ADDRESS"]
    DATE = ["DATE"]
    NUMBER = ["NUMBER"]
    PRICE = ["PRICE"]

    FACILITY = ["FACILITY"]
    BRAND = ["BRAND"]
    MOVIE = ["MOVIE"]
    MUSIC = ["MUSIC"]
    BOOK = ["BOOK"]
    SOFTWARE = ["SOFTWARE"]
    GAME = ["GAME"]
    PERSONAL_TITLE = ["PERSONALE_TITLE"]
    QUANTITY = ["QUANTITY"]
    ATTRIBUTE = ["ATTRIBUTE"]

    OTHER = ["OTHER"]
