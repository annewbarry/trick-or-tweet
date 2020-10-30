import pandas as pd
import numpy as np
import re
import nltk
import emoji
import spacy
import math
import string
import unicodedata
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from spellchecker import SpellChecker
from nltk.tokenize import sent_tokenize, word_tokenize
from langid.langid import LanguageIdentifier, model

sw = set(stopwords.words('english'))
new_words = set([i.lower() for i in ['sprintcare', 'Ask_Spectrum', 'VerizonSupport', 'ChipotleTweets',
       'AskPlayStation', 'marksandspencer', 'MicrosoftHelps',
       'ATVIAssist', 'AdobeCare', 'AmazonHelp', 'XboxSupport',
       'AirbnbHelp', 'AirAsiaSupport', 'Morrisons', 'NikeSupport',
       'AskAmex', 'YahooCare', 'AskLyft', 'UPSHelp', 'Delta', 'McDonalds',
       'AppleSupport', 'Uber_Support', 'Tesco', 'SpotifyCares',
       'British_Airways', 'comcastcares', 'AmericanAir', 'TMobileHelp',
       'VirginTrains', 'SouthwestAir', 'AskeBay', 'hulu_support',
       'GWRHelp', 'sainsburys', 'AskPayPal', 'HPSupport', 'ChaseSupport',
       'CoxHelp', 'DropboxSupport', 'VirginAtlantic', 'BofA_Help',
       'AzureSupport', 'AlaskaAir', 'ArgosHelpers', 'Postmates_Help',
       'AskTarget', 'GoDaddyHelp', 'CenturyLinkHelp', 'AskPapaJohns',
       'SW_Help', 'nationalrailenq', 'askpanera', 'Walmart',
       'USCellularCares', 'AsurionCares', 'GloCare', 'idea_cares',
       'DoorDash_Help', 'NeweggService', 'VirginAmerica',
       'Ask_WellsFargo', 'O2', 'asksalesforce', 'airtel_care', 'Kimpton',
       'AskCiti', 'IHGService', 'JetBlue', 'BoostCare', 'JackBox',
       'HiltonHelp', 'GooglePlayMusic', 'KFC_UKI_Help', 'DellCares',
       'TwitterSupport', 'GreggsOfficial', 'LondonMidland', 'ATT',
       'TacoBellTeam', 'Safaricom_Care', 'AskRBC', 'ArbysCares',
       'NortonSupport', 'AskSeagate', 'sizehelpteam', 'TfL', 'AldiUK',
       'SCsupport', 'AskDSC', 'AskVirginMoney', 'AskRobinhood',
       'MTNC_Care', 'DunkinDonuts', 'AWSSupport', 'VMUcare',
       'mediatemplehelp', 'MOO', 'PandoraSupport', 'askvisa',
       'OPPOCareIN', 'ask_progressive', 'PearsonSupport', 'AskTigogh',
       'OfficeSupport', 'CarlsJr', 'HotelTonightCX', 'KeyBank_Help']])
sw.update(new_words)

wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
lemmatizer = WordNetLemmatizer()
# Provided by https://github.com/NeelShah18/emot/blob/master/emot/emo_unicode.py
EMOTICONS = {
    u":‑\)":"Happy face or smiley",
    u":\)":"Happy face or smiley",
    u":-\]":"Happy face or smiley",
    u":\]":"Happy face or smiley",
    u":-3":"Happy face smiley",
    u":3":"Happy face smiley",
    u":->":"Happy face smiley",
    u":>":"Happy face smiley",
    u"8-\)":"Happy face smiley",
    u":o\)":"Happy face smiley",
    u":-\}":"Happy face smiley",
    u":\}":"Happy face smiley",
    u":-\)":"Happy face smiley",
    u":c\)":"Happy face smiley",
    u":\^\)":"Happy face smiley",
    u"=\]":"Happy face smiley",
    u"=\)":"Happy face smiley",
    u":‑D":"Laughing, big grin or laugh with glasses",
    u":D":"Laughing, big grin or laugh with glasses",
    u"8‑D":"Laughing, big grin or laugh with glasses",
    u"8D":"Laughing, big grin or laugh with glasses",
    u"X‑D":"Laughing, big grin or laugh with glasses",
    u"XD":"Laughing, big grin or laugh with glasses",
    u"=D":"Laughing, big grin or laugh with glasses",
    u"=3":"Laughing, big grin or laugh with glasses",
    u"B\^D":"Laughing, big grin or laugh with glasses",
    u":-\)\)":"Very happy",
    u":‑\(":"Frown, sad, andry or pouting",
    u":-\(":"Frown, sad, andry or pouting",
    u":\(":"Frown, sad, andry or pouting",
    u":‑c":"Frown, sad, andry or pouting",
    u":c":"Frown, sad, andry or pouting",
    u":‑<":"Frown, sad, andry or pouting",
    u":<":"Frown, sad, andry or pouting",
    u":‑\[":"Frown, sad, andry or pouting",
    u":\[":"Frown, sad, andry or pouting",
    u":-\|\|":"Frown, sad, andry or pouting",
    u">:\[":"Frown, sad, andry or pouting",
    u":\{":"Frown, sad, andry or pouting",
    u":@":"Frown, sad, andry or pouting",
    u">:\(":"Frown, sad, andry or pouting",
    u":'‑\(":"Crying",
    u":'\(":"Crying",
    u":'‑\)":"Tears of happiness",
    u":'\)":"Tears of happiness",
    u"D‑':":"Horror",
    u"D:<":"Disgust",
    u"D:":"Sadness",
    u"D8":"Great dismay",
    u"D;":"Great dismay",
    u"D=":"Great dismay",
    u"DX":"Great dismay",
    u":‑O":"Surprise",
    u":O":"Surprise",
    u":‑o":"Surprise",
    u":o":"Surprise",
    u":-0":"Shock",
    u"8‑0":"Yawn",
    u">:O":"Yawn",
    u":-\*":"Kiss",
    u":\*":"Kiss",
    u":X":"Kiss",
    u";‑\)":"Wink or smirk",
    u";\)":"Wink or smirk",
    u"\*-\)":"Wink or smirk",
    u"\*\)":"Wink or smirk",
    u";‑\]":"Wink or smirk",
    u";\]":"Wink or smirk",
    u";\^\)":"Wink or smirk",
    u":‑,":"Wink or smirk",
    u";D":"Wink or smirk",
    u":‑P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"X‑P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"XP":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":‑Þ":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":Þ":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":b":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"d:":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"=p":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u">:P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":‑/":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":/":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":-[.]":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u">:[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u">:/":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u"=/":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u"=[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":L":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u"=L":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":S":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":‑\|":"Straight face",
    u":\|":"Straight face",
    u":$":"Embarrassed or blushing",
    u":‑x":"Sealed lips or wearing braces or tongue-tied",
    u":x":"Sealed lips or wearing braces or tongue-tied",
    u":‑#":"Sealed lips or wearing braces or tongue-tied",
    u":#":"Sealed lips or wearing braces or tongue-tied",
    u":‑&":"Sealed lips or wearing braces or tongue-tied",
    u":&":"Sealed lips or wearing braces or tongue-tied",
    u"O:‑\)":"Angel, saint or innocent",
    u"O:\)":"Angel, saint or innocent",
    u"0:‑3":"Angel, saint or innocent",
    u"0:3":"Angel, saint or innocent",
    u"0:‑\)":"Angel, saint or innocent",
    u"0:\)":"Angel, saint or innocent",
    u":‑b":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"0;\^\)":"Angel, saint or innocent",
    u">:‑\)":"Evil or devilish",
    u">:\)":"Evil or devilish",
    u"\}:‑\)":"Evil or devilish",
    u"\}:\)":"Evil or devilish",
    u"3:‑\)":"Evil or devilish",
    u"3:\)":"Evil or devilish",
    u">;\)":"Evil or devilish",
    u"\|;‑\)":"Cool",
    u"\|‑O":"Bored",
    u":‑J":"Tongue-in-cheek",
    u"#‑\)":"Party all night",
    u"%‑\)":"Drunk or confused",
    u"%\)":"Drunk or confused",
    u":-###..":"Being sick",
    u":###..":"Being sick",
    u"<:‑\|":"Dump",
    u"\(>_<\)":"Troubled",
    u"\(>_<\)>":"Troubled",
    u"\(';'\)":"Baby",
    u"\(\^\^>``":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    u"\(\^_\^;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    u"\(-_-;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    u"\(~_~;\) \(・\.・;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    u"\(-_-\)zzz":"Sleeping",
    u"\(\^_-\)":"Wink",
    u"\(\(\+_\+\)\)":"Confused",
    u"\(\+o\+\)":"Confused",
    u"\(o\|o\)":"Ultraman",
    u"\^_\^":"Joyful",
    u"\(\^_\^\)/":"Joyful",
    u"\(\^O\^\)／":"Joyful",
    u"\(\^o\^\)／":"Joyful",
    u"\(__\)":"Kowtow as a sign of respect, or dogeza for apology",
    u"_\(\._\.\)_":"Kowtow as a sign of respect, or dogeza for apology",
    u"<\(_ _\)>":"Kowtow as a sign of respect, or dogeza for apology",
    u"<m\(__\)m>":"Kowtow as a sign of respect, or dogeza for apology",
    u"m\(__\)m":"Kowtow as a sign of respect, or dogeza for apology",
    u"m\(_ _\)m":"Kowtow as a sign of respect, or dogeza for apology",
    u"\('_'\)":"Sad or Crying",
    u"\(/_;\)":"Sad or Crying",
    u"\(T_T\) \(;_;\)":"Sad or Crying",
    u"\(;_;":"Sad of Crying",
    u"\(;_:\)":"Sad or Crying",
    u"\(;O;\)":"Sad or Crying",
    u"\(:_;\)":"Sad or Crying",
    u"\(ToT\)":"Sad or Crying",
    u";_;":"Sad or Crying",
    u";-;":"Sad or Crying",
    u";n;":"Sad or Crying",
    u";;":"Sad or Crying",
    u"Q\.Q":"Sad or Crying",
    u"T\.T":"Sad or Crying",
    u"QQ":"Sad or Crying",
    u"Q_Q":"Sad or Crying",
    u"\(-\.-\)":"Shame",
    u"\(-_-\)":"Shame",
    u"\(一一\)":"Shame",
    u"\(；一_一\)":"Shame",
    u"\(=_=\)":"Tired",
    u"\(=\^\·\^=\)":"cat",
    u"\(=\^\·\·\^=\)":"cat",
    u"=_\^=	":"cat",
    u"\(\.\.\)":"Looking down",
    u"\(\._\.\)":"Looking down",
    u"\^m\^":"Giggling with hand covering mouth",
    u"\(\・\・?":"Confusion",
    u"\(?_?\)":"Confusion",
    u">\^_\^<":"Normal Laugh",
    u"<\^!\^>":"Normal Laugh",
    u"\^/\^":"Normal Laugh",
    u"\（\*\^_\^\*）" :"Normal Laugh",
    u"\(\^<\^\) \(\^\.\^\)":"Normal Laugh",
    u"\(^\^\)":"Normal Laugh",
    u"\(\^\.\^\)":"Normal Laugh",
    u"\(\^_\^\.\)":"Normal Laugh",
    u"\(\^_\^\)":"Normal Laugh",
    u"\(\^\^\)":"Normal Laugh",
    u"\(\^J\^\)":"Normal Laugh",
    u"\(\*\^\.\^\*\)":"Normal Laugh",
    u"\(\^—\^\）":"Normal Laugh",
    u"\(#\^\.\^#\)":"Normal Laugh",
    u"\（\^—\^\）":"Waving",
    u"\(;_;\)/~~~":"Waving",
    u"\(\^\.\^\)/~~~":"Waving",
    u"\(-_-\)/~~~ \($\·\·\)/~~~":"Waving",
    u"\(T_T\)/~~~":"Waving",
    u"\(ToT\)/~~~":"Waving",
    u"\(\*\^0\^\*\)":"Excited",
    u"\(\*_\*\)":"Amazed",
    u"\(\*_\*;":"Amazed",
    u"\(\+_\+\) \(@_@\)":"Amazed",
    u"\(\*\^\^\)v":"Laughing,Cheerful",
    u"\(\^_\^\)v":"Laughing,Cheerful",
    u"\(\(d[-_-]b\)\)":"Headphones,Listening to music",
    u'\(-"-\)':"Worried",
    u"\(ーー;\)":"Worried",
    u"\(\^0_0\^\)":"Eyeglasses",
    u"\(\＾ｖ\＾\)":"Happy",
    u"\(\＾ｕ\＾\)":"Happy",
    u"\(\^\)o\(\^\)":"Happy",
    u"\(\^O\^\)":"Happy",
    u"\(\^o\^\)":"Happy",
    u"\)\^o\^\(":"Happy",
    u":O o_O":"Surprised",
    u"o_0":"Surprised",
    u"o\.O":"Surpised",
    u"\(o\.o\)":"Surprised",
    u"oO":"Surprised",
    u"\(\*￣m￣\)":"Dissatisfied",
    u"\(‘A`\)":"Snubbed or Deflated"
}

# Provided by https://github.com/rishabhverma17/sms_slang_translator/blob/master/slang.txt
text_speech = """
AFAIK=As Far As I Know
AFK=Away From Keyboard
ASAP=As Soon As Possible
ATK=At The Keyboard
ATM=At The Moment
A3=Anytime, Anywhere, Anyplace
BAK=Back At Keyboard
BBL=Be Back Later
BBS=Be Back Soon
BFN=Bye For Now
B4N=Bye For Now
BRB=Be Right Back
BRT=Be Right There
BTW=By The Way
B4=Before
B4N=Bye For Now
CU=See You
CUL8R=See You Later
CYA=See You
FAQ=Frequently Asked Questions
FC=Fingers Crossed
FWIW=For What It's Worth
FYI=For Your Information
GAL=Get A Life
GG=Good Game
GN=Good Night
GMTA=Great Minds Think Alike
GR8=Great!
G9=Genius
IC=I See
ICQ=I Seek you (also a chat program)
ILU=ILU: I Love You
IMHO=In My Honest/Humble Opinion
IMO=In My Opinion
IOW=In Other Words
IRL=In Real Life
KISS=Keep It Simple, Stupid
LDR=Long Distance Relationship
LMAO=Laugh My A.. Off
LOL=Laughing Out Loud
LTNS=Long Time No See
L8R=Later
MTE=My Thoughts Exactly
M8=Mate
NRN=No Reply Necessary
OIC=Oh I See
PITA=Pain In The A..
PRT=Party
PRW=Parents Are Watching
QPSA?	Que Pasa?
ROFL=Rolling On The Floor Laughing
ROFLOL=Rolling On The Floor Laughing Out Loud
ROTFLMAO=Rolling On The Floor Laughing My A.. Off
SK8=Skate
STATS=Your sex and age
ASL=Age, Sex, Location
THX=Thank You
TTFN=Ta-Ta For Now!
TTYL=Talk To You Later
U=You
SMH = Shake My Head
U2=You Too
U4E=Yours For Ever
WB=Welcome Back
WTF=What The F...
WTG=Way To Go!
WUF=Where Are You From?
W8=Wait...
7K=Sick:-D Laugher
"""

chat_dict = {}
chat_list = []
for i in text_speech.split("\n"):
    if i != "":
        tw = i.split("=")[0]
        full_text = i.split("=")[-1]
        chat_list.append(tw)
        chat_dict[tw] = full_text
            
chat_list = set(chat_list)

def day_of_week_num(dts):
    '''
    weekday: takes in the day_of_week_num converted column
    and assigns those numbers (after looking up which day
    of the week 0 landed on) and assigned a string representation
    of that day of the week
    Parameters
    ----------
    dts: Integer
    Returns
    -------
    dts: Python string of the day of the week
    '''
    return (dts.view('int64') - 4) % 7

def weekday(dts):
    '''
    weekday: takes in the day_of_week_num converted column
    and assigns those numbers (after looking up which day
    of the week 0 landed on) and assigned a string representation
    of that day of the week
    Parameters
    ----------
    dts: Integer
    Returns
    -------
    dts: Python string of the day of the week
    '''
    if dts == 0:
        dts = 'Tuesday'
    elif dts == 1:
        dts = 'Wednesday'
    elif dts == 2:
        dts = 'Thursday'
    elif dts == 3:
        dts = 'Friday'
    elif dts == 4:
        dts = 'Saturday'
    elif dts == 5:
        dts = 'Sunday'
    else:
        dts = 'Monday'
    return dts

def set_response_category(col):
    '''
    set_response_category: takes in the minutes to respond
    and sets a cutoff point between less than or equal to 27
    as "Fast" and above 27 as "Slow"
    Parameters
    ----------
    doc: Numpy Array
    Returns
    -------
    doc: Python list of categorical values
    '''
    lst = []
    for i in col:
        if i <= 27.0:
            lst.append('Fast')
        else:
            lst.append('Slow')
    return lst


def dataframe_clean(df_file_path):
    '''
    dataframe_clean: takes in a string and converts
    emoticons ':), :(, :-), etc.' with word values.
    Parameters
    ----------
    doc: Python string
    Returns
    -------
    doc: Python string with cleaned dataframe
    '''
    tweets = pd.read_csv(df_file_path)
    company_responses = tweets[tweets['inbound'] == False]
    # Pick only inbound tweets that aren't in reply to anything...
    first_inbound = tweets[pd.isnull(tweets.in_response_to_tweet_id) & tweets.inbound]
    # Merge in all tweets in response
    inbounds_and_outbounds = pd.merge(first_inbound, tweets, left_on='tweet_id', 
                                      right_on='in_response_to_tweet_id')
    # Filter out cases where reply tweet isn't from company
    tweets = inbounds_and_outbounds[inbounds_and_outbounds.inbound_y ^ True]
    tweets = tweets.drop(['response_tweet_id_x', 'in_response_to_tweet_id_x', 
                          'response_tweet_id_y', 'in_response_to_tweet_id_y', 'tweet_id_x','tweet_id_y' ], axis = 1)
    # Converts date columns into datetime
    tweets.created_at_x = pd.to_datetime(tweets.created_at_x)
    tweets.created_at_y = pd.to_datetime(tweets.created_at_y)
    # Calculates time_to_respond by subtracting the time between customer and customer support team responses
    tweets['time_to_respond'] = tweets.created_at_y - tweets.created_at_x
    tweets = tweets.drop(['created_at_y','inbound_x','inbound_y'], axis = 1)
    tweets.columns = ['customer_tweet_id', 'time_tweeted', 'customer_tweet_text', 'company_name', 'company_response_text','time_to_respond']
    # Calculates which day of the week the tweet was made (Mon-Sun)
    tweets.time_tweeted = tweets.time_tweeted.apply(lambda x: day_of_week_num(x))
    tweets.time_tweeted = tweets.time_tweeted.apply(lambda x: weekday(x))
    tweets.rename({'time_tweeted':'day_tweeted'}, axis = 1, inplace = True)
    # Converts thetime to respond into minutes
    seconds = tweets['time_to_respond'] / np.timedelta64(1, 's')
    tweets.time_to_respond = seconds
    tweets.time_to_respond = tweets.time_to_respond.apply(lambda x: math.ceil(x/60))
    tweets.rename({'time_to_respond': 'minutes_to_respond'},axis = 1, inplace = True)
    # Sets the target values of the dataframe to be above 27 minutes and below or equal to 27 minutes
    tweets['Reponse_Speed'] = set_response_category(tweets.minutes_to_respond)
    return tweets

def lowercase(doc):
    '''
    lowercase: takes in a string and lowercases it
    Parameters
    ----------
    doc: Python string
    Returns
    -------
    doc: Python strin converted to lowercase
    '''
    doc = doc.lower()
    return doc


def remove_punctuation(doc):
    '''
    remove_punctuation: takes in a string and 
    revoes the punctuation via RegEx
    Parameters
    ----------
    doc: Python string
    Returns
    -------
    doc: Python string with punctuation removed
    '''
    return re.sub(r'[^\w\s]','',doc)



def remove_html(doc):
    '''
    remove_url: takes in a string and removes
    html values '<.X?>' using RegEx
    Parameters
    ----------
    doc: Python string
    Returns
    -------
    doc: Python string with removed html text
    '''
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'',doc)

def remove_url(doc):
    '''
    remove_url: takes in a string and removes
    url values 'https://, etc.' using RegEx
    Parameters
    ----------
    doc: Python string
    Returns
    -------
    doc: Python string with removed url text
    '''
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub(r'', doc)

def convert_emoticons(doc):
    '''
    convert_emoticons: takes in a string and converts
    emoticons ':), :(, :-), etc.' with word values.
    Parameters
    ----------
    doc: Python string
    Returns
    -------
    doc: Python string with converted emoticons
    '''
    for i in EMOTICONS:
        doc = re.sub(u'('+i+')', "_".join(EMOTICONS[i].replace(",","").split()), doc)
    return doc

def convert_emojis(doc):
    '''
    convert_emojis: takes in a string and converts
    emojis '😀, 😂, etc.' with word values.
    Parameters
    ----------
    doc: Python string
    Returns
    -------
    doc: Python string with converted emojis
    '''
    doc = emoji.demojize(doc)
    doc = re.sub('[:]', '', doc)
    return doc

def convert_text_speech(doc):
    '''
    convert_text_speech: takes in a string and converts any
    abbreviated text speech, 'ttyl, imo, wtf, etc.' and converts
    it into its component words
    Parameters
    ----------
    doc: Python string
    Returns
    -------
    doc: Python string with converted text speech
    '''
    text = []
    #removes punctuation because that throws off the conversion
    doc = doc.translate(str.maketrans('', '', string.punctuation))
    for i in doc.split():
        if i.upper() in chat_list:
            text.append(chat_dict[i.upper()])
        else:
            text.append(i)
    return " ".join(text)


def remove_stopwords(doc):
    '''
    remove_stopwords: takes in a list of strings removes
    those strings that contain stopwords imported from the 
    nlkt.corpus library
    Parameters
    ----------
    doc: Python string
    Returns
    -------
    doc: Python list with stop words removed
    '''
    doc = [word for word in doc if word not in sw]
    return doc

def set_english(df, text_column):
    '''
    set_english: takes in a string of a text column
    outputs the predicted languge of that string to a list
    which is used to mask a pandas DataFrame.
    Parameters
    ----------
    doc: Python string
    Returns
    -------
    doc: Python list with predicted language
    '''
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)  
    lst =[]
    for i in df[text_column].values: 
        lang, score = identifier.classify(i)
        lst.append(lang)
    return lst

def lemmatize_words(doc):
    '''
    lemmatize_words: takes in a list of strings and
    outputs the lemmatized version of each word,
    lemmatizes based on part of speech.
    Parameters
    ----------
    doc: Python string
    Returns
    -------
    doc: Python list with lemmatized words
    '''
    pos_tagged_text = nltk.pos_tag(doc)
    doc = [lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text]
    return doc

def remove_numbers(doc):
    '''
    remove_numbers: removes all numbers from a string
    Parameters
    ----------
    doc: Python string
    Returns
    -------
    doc: Python list with numbers removed
    '''
    doc = re.sub(r'\w*\d\w*', '', doc).strip()
    return doc

def remove_non_english_characters(doc):
    '''
    remove_non_english_characters: takes in a list of strings and
    removes all non-english characters, including ASCII.
    Parameters
    ----------
    doc: Python string
    Returns
    -------
    doc: Python list with non-English characters removed
    '''
    doc = unicodedata.normalize('NFKD', doc).encode('ASCII', 'ignore').decode('utf8')
    return doc


def preprocessing(doc):
    '''
    preprocessing: takes in a string and applies 
    text preprocessing to that string. 
    (For each function definition, refer to above.)
    Parameters
    ----------
    doc: Python string
    Returns
    -------
    doc: Python list with preprocessed text
    '''
    doc = lowercase(doc)
    doc = remove_punctuation(doc)
    doc = convert_emoticons(doc)
    doc = convert_emojis(doc)
    doc = remove_url(doc)
    doc = remove_html(doc)
    doc = convert_text_speech(doc)
    doc = remove_numbers(doc)
    doc = remove_non_english_characters(doc)
    doc = word_tokenize(doc)
    doc = remove_stopwords(doc)
    doc = lemmatize_words(doc)
    return doc