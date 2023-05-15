""" Utility classes and functions related to SwitchPrompt (EACL 2023).
Copyright (c) 2022 Robert Bosch GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
#nltk.data.path.append('/home/kgo2rng/nltk_data/nltk_data')

from collections import Counter, defaultdict


class DomainScorer:
    
    def __init__(self, 
                 general_sentences, 
                 domain_sentences, 
                 domain_documents=None, 
                 lowercase=True, 
                 alpha=-1e5, 
                 beta=1e2, 
                 transformer=None, 
                 transformer_tokenizer=None):
        
        # Compute term-frequencies (tf) and inverse-document-frequencies (idf)
        self.tf_general = self.tf(general_sentences, lowercase=lowercase)
        self.tf_domain = self.tf(domain_sentences, lowercase=lowercase)
        if domain_documents is None:
            self.idf_domain = self.idf(domain_sentences, lowercase=lowercase)
        else:
            self.idf_domain = self.idf(domain_documents, lowercase=lowercase)
            for key, value in self.tf(domain_documents, lowercase=lowercase).items():
                if key in self.tf_domain:
                    self.tf_domain[key] += value
                else:
                    self.tf_domain[key] = value
        
        # Get and add stopwords
        self._stopwords = set(stopwords.words('english'))
        self._stopwords.add("'s")
        self._stopwords.add("n't")
        
        # Set lowercasing options
        if lowercase:
            self._stopwords = [s.lower() for s in self._stopwords]
        self.lowercase = lowercase
        
        # Set tf-idf hyperparameters and extract static domain keywords
        self.alpha = alpha
        self.beta = beta
        self.static_keywords = self.get_dynamic_keywords(
            ' '.join([' '.join([token for token in sent]) for sent in domain_sentences]), 
            k=25,
        )
        
        # If available, compute BERT-embeddings of static keywords
        if transformer is not None and transformer_tokenizer is not None:
            self.transformer = transformer
            self.transformer_tokenizer = transformer_tokenizer
            #self.keyword_embeddings = np.zeros((
            #    len(self.static_keywords), 
            #    self.transformer.config.hidden_size
            #))
            keyword_embeddings = []
            for keyword in self.static_keywords:
                embedding = self._get_cls_embedding(keyword)
                keyword_embeddings.append(embedding)
            self.keyword_embeddings = np.array(keyword_embeddings)
        else:
            self.transformer = None
            self.transformer_tokenizer = None
            
    @staticmethod
    def tf(questions, smoothing_factor=1, normalize=True, lowercase=True): 
        """
        Estimates the term frequency on a given corpus: 
        
        term_frequency = number of times a given term appears in document
        """
        term_freqs = Counter()
        for q in questions:
            for t in q:
                term_freqs[t.lower() if lowercase else t] += 1

        max_freq = -1
        if smoothing_factor != 0 or normalize:
            for t in term_freqs:
                term_freqs[t] += smoothing_factor
                max_freq = max(max_freq, term_freqs[t])
        if normalize:
            term_freqs_norm = defaultdict(lambda: 1e-5)
            for t in term_freqs:
                term_freqs_norm[t] = term_freqs[t] / max_freq
            return term_freqs_norm
        else:
            term_freqs_smoothed = defaultdict(lambda: smoothing_factor)
            for t in term_freqs:
                term_freqs_smoothed[t] = term_freqs[t]
            return term_freqs_smoothed

    @staticmethod
    def idf(questions, smoothing_factor=1, lowercase=True): 
        """
        Estimates the inverse document frequency on a domain-specific corpus:
        
        inverse_document_frequency = log(total number of documents / number of documents with term) + 1

        We use a version with smoothing that adds a "1" to the numerator and denominator:

        inverse_document_frequency = log((1 + total_number_of_documents) / (number_of_documents_with_term +1)) + 1
        """
        docs_with_term = Counter()
        for q in questions:
            for t in q:
                docs_with_term[t.lower() if lowercase else t] += 1

        idf = defaultdict(lambda: 1e-5)
        for t in docs_with_term:
            number_of_documents_with_term = docs_with_term[t] + smoothing_factor
            number_of_documents = len(docs_with_term) + smoothing_factor
            idf[t] = np.log(number_of_documents / number_of_documents_with_term) + 1
        return idf


    def domain_specificness(self, term, alpha=None, beta=None):
        """
        Estimates the domain specificness of a term
        
        domain_specificness = alpha * tf_general + beta * tf_domain + idf_doman
        
        We use alpha (<= 0) and beta (>= 0) for weighting the 
        genral and domain-specific term frequency, respectively.
        
        In practice, we found alpha=-1e5 and beta=1e2 work well. 
        This penalizes general domain rather high and boosts domain-specific terms. 
        """
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            beta = self.beta
        score = alpha * self.tf_general[term] +  beta * self.tf_domain[term] + self.idf_domain[term]
        return score
    
    @staticmethod
    def _select_top_k_keywords(tokens, scores, k=3):
        keywords = []
        for i in range(min(len(tokens), k)):
            best_idx = np.argmax(scores)
            keywords.append(tokens[best_idx])
            del scores[best_idx]
            del tokens[best_idx]
        return keywords
    
    @staticmethod
    def _get_selected_tokens(tokens, keywords, lowercased):
        """ Keywords might be lowercased; get the "real" tokens """
        output = []
        for k in keywords:
            for t in tokens:
                if k == t or (lowercased and k == t.lower()):
                    output.append(t)
                    break
        return output
    
    def _filter_tokens(self, tokens, filter_pos_tags, filter_stopwords):
        tokens = [t for t in tokens]
        if filter_pos_tags:
            pos_tags = nltk.pos_tag(tokens)
            tokens = [t for t, (_, p) in zip(tokens, pos_tags) if p[0] in [
                'N', # Nouns
                #'V', # Verbs
                'R', # Adverbs
                'J', # Adjectives
            ]]
            # See https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
        if self.lowercase:
            tokens = [t.lower() for t in tokens]
        if filter_stopwords:
            tokens = [t for t in tokens if t not in self._stopwords and t not in string.punctuation]
        ##added as we do not require only numbers as keywords    
        tokens = [t for t in tokens if  not t.isdigit()]
        return tokens
    
    def _get_cls_embedding(self, input_text): 
        tokenized_input = self.transformer_tokenizer(input_text, return_tensors="pt")
        tokenized_input = tokenized_input.to(self.transformer.device)
        self.transformer.eval()
        outputs = self.transformer(**tokenized_input)
        cls_embedding = outputs.last_hidden_state[0][0]
        cls_embedding = cls_embedding.cpu().detach().numpy()
        return cls_embedding
    
    def get_dynamic_keywords(self, input_tokens, k=3, alpha=None, beta=None, filter_pos_tags=False, filter_stopwords=True):
        if isinstance(input_tokens, str): 
            input_tokens = nltk.word_tokenize(input_tokens)
        tokens = self._filter_tokens(input_tokens, filter_pos_tags, filter_stopwords)
        tokens = list(set(tokens))
        scores = [self.domain_specificness(t, alpha, beta) for t in tokens]
        keywords = self._select_top_k_keywords(tokens, scores, k)
        keywords = self._get_selected_tokens(input_tokens, keywords, self.lowercase)
        return keywords
        
    def get_static_keywords(self, k=3):
        return self.static_keywords[:k]
    
    def get_mixed_keywords(self, input_tokens, k_s=3, k_d=3, k=6, alpha=None, beta=None, filter_pos_tags=False, filter_stopwords=True):
        if isinstance(input_tokens, str): 
            input_tokens = nltk.word_tokenize(input_tokens)
        if k_s is None or k_d is None:
            if k % 2 == 0:
                k_s, k_d = int(k/2), int(k/2)
            else:
                k_s, k_d = int(k/2)+1, int(k/2)
        dynamic_keywords = self.get_dynamic_keywords(input_tokens, k_d, alpha, beta, filter_pos_tags, filter_stopwords)
        static_keywords = self.get_static_keywords(k_s)
        return static_keywords + dynamic_keywords
    
    def get_most_similar_keywords(self, input_tokens, k=3): 
        if self.transformer is None:
            return []
        if isinstance(input_tokens, str): 
            input_tokens = nltk.word_tokenize(input_tokens)
        
        x = self._get_cls_embedding(' '.join(input_tokens))
        M = self.keyword_embeddings
        
        # cosine similarity
        dot_product = np.dot(x, M.T)
        norm_a = np.linalg.norm(x)
        norm_b = np.linalg.norm(M, axis=1)
        score = dot_product / (norm_a * norm_b)
            
        # print(self.static_keywords)
        # print(list(score))
        # print(k)
        
        similar_keywords = self._select_top_k_keywords(list(self.static_keywords), list(score), k)
        return similar_keywords