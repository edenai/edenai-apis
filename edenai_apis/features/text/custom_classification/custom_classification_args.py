# pylint: disable=locally-disabled, line-too-long
def custom_classification_arguments(provider_name: str):
    return {
        "texts": [
            "Confirm your email address",
            "hey i need u to send some $",
            "Congratulations! You've won a free vacation! Click the link to claim your prize now!",
            "Make money fast! Join our exclusive program and start earning thousands in just a few days!",
            "Get rich quick with this amazing investment opportunity. Guaranteed returns!",
            "Unlock special discounts on luxury goods. Limited-time offer! Click here to shop now!",
            "You've been selected for a special promotion. Act now to secure your spot!",
            "Meet hot singles in your area! Chat now and find your perfect match!",
            "Eliminate debt effortlessly! Our program can erase your financial worries.",
            "Claim your inheritance! Just provide your bank details for a seamless transfer.",
            "Reminder: Your appointment with Dr. Smith is scheduled for tomorrow at 2:00 PM.",
            "Update: The event you registered for has been rescheduled to [New Date].",
            "Welcome to our newsletter! Stay tuned for updates on our latest products and promotions.",
        ],
        "labels": ["spam", "not spam"],
        "examples": [
            ["I need help please wire me $1000 right now", "spam"],
            ["Dermatologists dont like her!", "spam"],
            ["Please help me?", "spam"],
            ["Pre-read for tomorrow", "not spam"],
            ["Your parcel will be delivered today", "not spam"],
            ["Review changes to our Terms and Conditions", "not spam"],
        ],
        "settings": {"openai": "gpt-4o", "xai": "grok-2-latest"},
    }
