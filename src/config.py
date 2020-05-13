
max_utterance_length = 15

categories = [
            'FEEDBACK', 'GREET', 'INFORM', 'SUGGEST', 'INIT', 'CLOSE', 'REQUEST', 'DELIBERATE', 'BYE', 'COMMIT',
            'THANK', 'POLITENESS_FORMULA', 'BACKCHANNEL', 'INTRODUCE', 'DEFER', 'OFFER', 'UNK'
             ]

most_freq_categories = [ 'FEEDBACK',  'INFORM', 'SUGGEST', 'REQUEST','DELIBERATE', 'BACKCHANNEL']



french_categories = ['A','QY','QO','D','RO','N','I','B','G','R','Y','S']


##based on switch corpus which was predicted
most_freq_predicted_categories = ['FEEDBACK',  'INFORM', 'SUGGEST', 'REQUEST', 'DELIBERATE', 'BACKCHANNEL']


switchboard_categories = [
'sd','b','sv','aa','%', 'ba','qy','x','ny','fc','%','qw','nn','bk','h','qy^d','fo_o_fw_"_by_bc','bh','^q','bf',
'na','ad','^2','b^m','qo','qh','^h','ar','ng','br','no','fp','qrr','arp_nd','t3','oo_co_cc','t1','bd','aap_am',
'^g','qw^d','fa','ft'

]

english_embedding_path = "target-space-en.no-trans"
german_embedding_path = "target-space-de.no-trans"
de_to_en_embedding_space = "source-space-from-de-to-en.CCA"
en_to_de_embedding_space = "source-space-from-en-to-de.CCA"

filter_uh = True

learning_rate = 0.002

additional_training_learning_rate = 0.001

w2v_dimension = 300
bert_dimension = 768

we_word_count = 300001

vocabulary_size = 20000

PADDING_SYMBOL = "%%PADDING%%"
OUT_OF_VOCABULARY_SYMBOL = "%%OUTOFVOCABULARY%%"


# A Twitter Corpus and Benchmark Resources for German Sentiment Analysis. by Mark Cieliebak, Jan Deriu, Fatih Uzdilli, and Dominic Egger. In “Proceedings of the 4th International Workshop on Natural Language Processing for Social Media (SocialNLP 2017)”, Valencia, Spain, 2017


