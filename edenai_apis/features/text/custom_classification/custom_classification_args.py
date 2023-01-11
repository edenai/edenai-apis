# pylint: disable=locally-disabled, line-too-long
def custom_classification_arguments():
    return {
        'texts' : [
            "Confirm your email address",
            "hey i need u to send some $"
            ],
        'labels' : [
            'spam',
            'not spam'
            ],
        'examples' : [
            ['I need help please wire me $1000 right now','spam'],
            ['Dermatologists dont like her!','spam'],
            ['Please help me?','spam'],
            ['Pre-read for tomorrow','not spam'],
            ['Your parcel will be delivered today','not spam'],
            ['Review changes to our Terms and Conditions','not spam'],
                      ]
    }
