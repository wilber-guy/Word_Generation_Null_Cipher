# Word_Generation_Null_Cipher
Using GPT-2, take in a secret word and generate a novel paragraph where each word's first letter spells the secret message. This form of secret hiding
(steganography) prevents anyone from knowing a secret even exists in the novel message. Thus making it more difficult to try and crack. This could be layered with a traditional encryption method to create a strongly encrypted message within another message. Doing this means an adversary could spend lots of time and resources trying to crack a code that is not there or assume there is no code in the message at all.

The model needs a starter sentence that will be used to start generating the novel text containing the secret. Longer secrets tend 
to become run-on sentences or gibberish quickly. A word or two can be hidden with relatively moderate success if you run the model a couple of times.
