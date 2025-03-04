class SubCategoryPattern:
    """This class contains all the patterns for the subcategories of the moderation category.

    Subclasses:
        * Toxic: This class contains a constant for each subcategory of the toxic category.
        * Content: This class contains a constant for each subcategory of the content category.
        * Sexual: This class contains a constant for each subcategory of the sexual category.
        * Violence: This class contains a constant for each subcategory of the violence category.
        * DrugAndAlcohol: This class contains a constant for each subcategory of the drugAndAlcohol category.
        * Finance: This class contains a constant for each subcategory of the finance category.
        * HateAndExtremism: This class contains a constant for each subcategory of the hateAndExtremism category.
        * Safe: This class contains a constant for each subcategory of the safe category.
        * Other: This class contains a constant for each subcategory of the other category.
    """

    class Toxic:
        """This class contains all the patterns for the subcategories of the moderation category.

        Constants:
            INSULT (list): List all the patterns for the insult subcategory.
            OBSCENE (list): List all the patterns for the obscene subcategory.
            DEROGATORY (list): List all the patterns for the derogatory subcategory.
            PROFANITY (list): List all the patterns for the profanity subcategory.
            THREAT (list): List all the patterns for the threat subcategory.
            TOXIC (list): List all the patterns for the toxic subcategory.
        """

        INSULT = ["insult"]
        OBSCENE = ["obscene", "obscene_gesture_content"]
        DEROGATORY = ["derogatory"]
        PROFANITY = ["profanity"]
        THREAT = ["threat"]
        TOXIC = ["toxic", "severe_toxic"]

    class Content:
        """This class contains all the patterns for the subcategories of the moderation category.

        Constants:
            MIDDLE_FINGER (list): List all the patterns for the middle finger subcategory.
            PUBLIC_SAFETY (list): List all the patterns for the public safety subcategory.
            HEALTH (list): List all the patterns for the health subcategory.
            EXPLICIT (list): List all the patterns for the explicit subcategory.
            QRCODE (list): List all the patterns for the qrcode subcategory.
            MEDICAL (list): List all the patterns for the medical subcategory.
            POLITICS (list): List all the patterns for the politics subcategory.
            LEGAL (list): List all the patterns for the legal subcategory.
        """

        MIDDLE_FINGER = ["middle finger"]
        PUBLIC_SAFETY = ["public safety"]
        HEALTH = ["health"]
        EXPLICIT = ["explicit"]
        QRCODE = ["qr_code_content"]
        MEDICAL = ["medical"]
        POLITICS = ["politics"]
        LEGAL = ["legal"]

    class Sexual:
        """This class contains all the patterns for the subcategories of the moderation category.

        Constants:
            SEXUAL (list): List all the patterns for the sexual subcategory.
            MINORS (list): List all the patterns for the minors subcategory.
            SEXUAL_ACTIVITY (list): List all the patterns for the sexual activity subcategory.
            SEXUAL_SITUATIONS (list): List all the patterns for the sexual situations subcategory.
            NUDITY (list): List all the patterns for the nudity subcategory.
            PARTIAL_NUDITY (list): List all the patterns for the partial nudity subcategory.
            SUGGESTIVE (list): List all the patterns for the suggestive subcategory.
            ADULT_TOYS (list): List all the patterns for the adult toys subcategory.
            REVEALING_CLOTHES (list): List all the patterns for the revealing clothes subcategory.
        """

        SEXUAL = ["sexual", "porn_content", "adult", "sexually explicit"]
        MINORS = ["sexual/minors"]
        SEXUAL_ACTIVITY = ["sexual activity"]
        SEXUAL_SITUATIONS = ["sexual situations"]
        NUDITY = [
            "nudity",
            "graphic male nudity",
            "graphic female nudity",
            "illustrated explicit nudity",
        ]
        PARTIAL_NUDITY = ["partial nudity", "barechested male"]
        SUGGESTIVE = [
            "female swimwear or underwear",
            "male swimwear or underwear",
            "suggestive",
            "suggestive_nudity_content",
            "sexually suggestive",
        ]
        ADULT_TOYS = ["adult toys"]
        REVEALING_CLOTHES = ["revealing clothes"]

    class Violence:
        """This class contains all the patterns for the subcategories of the moderation category.

        Constants:
            GRAPHIC_VIOLENCE_OR_GORE (list): List all the patterns for the graphic violence or gore subcategory.
            PHYSICAL_VIOLENCE (list): List all the patterns for the physical violence subcategory.
            WEAPON_VIOLENCE (list): List all the patterns for the weapon violence subcategory.
            VIOLENCE (list): List all the patterns for the violence subcategory.
        """

        GRAPHIC_VIOLENCE_OR_GORE = [
            "death, harm & tragedy",
            "violence/graphic",
            "graphic violence or gore",
            "gore",
            "gore_content",
            "emaciated bodies",
            "corpses",
            "hanging",
            "air crash",
            "explosions and blasts",
        ]
        PHYSICAL_VIOLENCE = [
            "self-harm",
            "self-harm/intent",
            "self-harm/instructions",
            "physical violence",
            "self injury",
        ]
        WEAPON_VIOLENCE = [
            "firearms & weapons",
            "war & conflict",
            "weapon violence",
            "weapons",
            "weapon_content",
        ]
        VIOLENCE = ["violent", "violence"]

    class DrugAndAlcohol:
        """This class contains all the patterns for the subcategories of the moderation category.

        Constants:
            DRUG_PRODUCTS (list): List all the patterns for the drug products subcategory.
            DRUG_USE (list): List all the patterns for the drug use subcategory.
            TOBACCO (list): List all the patterns for the tobacco subcategory.
            SMOKING (list): List all the patterns for the smoking subcategory.
            ALCOHOL (list): List all the patterns for the alcohol subcategory.
            DRINKING (list): List all the patterns for the drinking subcategory.
        """

        DRUG_PRODUCTS = [
            "drug products",
            "pills",
            "drug paraphernalia",
            "drug",
            "drug_content",
            "illicit drugs",
        ]
        DRUG_USE = ["drug use"]
        TOBACCO = ["tobacco products"]
        SMOKING = ["smoking"]
        ALCOHOL = ["alcoholic beverages"]
        DRINKING = ["drinking"]

    class Finance:
        """This class contains all the patterns for the subcategories of the moderation category.

        Constants:
            GAMBLING (list): List all the patterns for the gambling subcategory.
            MONEY_CONTENT (list): List all the patterns for the money content subcategory.
            FINANCE (list): List all the patterns for the finance subcategory.
        """

        GAMBLING = ["gambling"]
        MONEY_CONTENT = ["money_content"]
        FINANCE = ["finance"]

    class HateAndExtremism:
        """This class contains all the patterns for the subcategories of the moderation category.

        Constants:
            HATE (list): List all the patterns for the hate subcategory.
            HARASSMENT (list): List all the patterns for the harassment subcategory.
            THREATENING (list): List all the patterns for the threatening subcategory.
            EXTREMIST (list): List all the patterns for the extremist subcategory.
            RACY (list): List all the patterns for the racy subcategory.
        """

        HATE = ["identity_hate", "hate", "hate_sign_content"]
        HARASSMENT = ["harassment"]
        THREATENING = ["hate/threatening", "harassment/threatening"]
        EXTREMIST = ["extremist", "nazi party", "white supremacy"]
        RACY = ["racy"]

    class Safe:
        """This class contains all the patterns for the subcategories of the moderation category.

        Constants:
            NOT_SAFE (list): List all the patterns for the not safe subcategory.
            SAFE (list): List all the patterns for the safe subcategory.
        """

        NOT_SAFE = ["nsfw", "unsafe"]
        SAFE = ["safe", "sfw"]

    class Other:
        """This class contains all the patterns for the subcategories of the moderation category.

        Constants:
            SPOOF (list): List all the patterns for the spoof subcategory.
            RELIGION (list): List all the patterns for the religion subcategory.
            OFFENSIVE (list): List all the patterns for the offensive subcategory.
            OTHER (list): List all the patterns for the other subcategory.
        """

        SPOOF = ["spoof"]
        RELIGION = ["religion & belief"]
        OFFENSIVE = ["offensive"]
        OTHER = ["other"]
