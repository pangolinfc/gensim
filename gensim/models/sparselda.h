#ifndef SPARSELDA_H
#define SPARSELDA_H

/*
 * Everything in this library returns by value.  Functions that allocate 
 * memory will return 1 on success or 0 on failure.
 */

/*
 * A document, together with its latent parameters.
 *
 * I don't know what possessed me to make this a linked list.  I guess I had
 * functional data structures on the brain.
 */
typedef struct 
FeatureSequence
{
    // The next document, or null
    struct FeatureSequence *next;
    // The number of tokens in this document
    unsigned int length;
    // The list of tokens
    unsigned int *tokens;
    // The list of topic assignments
    unsigned int *topics;
    // The number of tokens assigned to each topic
    unsigned int *Ntd;
} FeatureSequence;

/* 
 * Holds & caches various parameters used by SparseLDA.
 *
 * Note that SparseLDA stores its word-topic matrix as a sparse matrix 
 * as follows:
 *      Nwt[w] = [(count, topic), (count, topic), ... ]
 *
 * sorted descending by the count.  The tuples are packed into a single 32-bit
 * unsigned integer by placing the topic in the lower Tbits bits, and the 
 * count in the upper (32 - Tbits) bits.
 *
 * Right now the library is not set up to shrink the sparse rows to the number
 * of non-zero elements in the vectors.  But some day I should probably do
 * that.
 */
typedef struct 
TopicModelParams
{
    // Number of topics
    unsigned int T;
    // Used to mask out the topic assignment
    unsigned int Tmask;
    // Number of bits used for topic assignment
    unsigned int Tbits;
    // Size of the vocabulary
    unsigned int V;
    // Number of documents
    unsigned int D;
    // Number of tokens
    unsigned int N;
    // Alpha vector (prior on topics in documents)
    double *alpha;
    // Constant beta (prior on words in topics)
    double beta;
    // Word-topic matrix
    unsigned int **Nwt;
    // Number of tokens assigned to each topic
    unsigned int *Nt;

    // Histograms for optimizing alpha
    unsigned int max_len;
    unsigned int *Cn;
    unsigned int **Ctn;

    // Scratch buffers
    double *denoms;
    double *q_;

    // And the documents
    FeatureSequence *documents;
} TopicModelParams;

/*
 * Free the memory used in the FeatureSequence data structure
 */
void
free_feature_sequence(FeatureSequence *fs);

/*
 * Free the memory used in the FeatureSequence data structure and all of its
 * children
 */
void
free_feature_sequence_list(FeatureSequence *fs);

/*
 * Free the memory used in the TopicModelParams data structure and its children
 */
void
free_topic_model_params(TopicModelParams *tmp);

/*
 * Allocate the memory for a topic model with T topics
 */
int
new_topic_model_params(TopicModelParams *tmp, unsigned int T);

/*
 * Initialize a FeatureSequence to random assignments and update the
 * TopicModelParams accordingly.
 */
int
init_feature_sequence(
        TopicModelParams *tmp, FeatureSequence *fs, unsigned int length);

/*
 * Add a document to the topic model and update the parameters accordingly
 */
int 
add_document(
        TopicModelParams *tmp, unsigned int *tokens, unsigned int length);

/*
 * Do one pass on the specified document
 */
void
do_document(FeatureSequence fs, TopicModelParams tmp);

/*
 * Do one pass on every document
 */
void
do_iteration(TopicModelParams *tmp);

/*
 * Fill topics[][] with the word-topic matrix so that
 *      topics[t][w] 
 * is the probability of word w being drawn from topic t
 */
void
sample_topics(TopicModelParams tmp, double **topics);

/*
 * Fill topics[][] with the word-topic matrix so that
 *      topics[t*T + w] = 
 * is the probability of word w being drawn from topic t
 */
void
flat_sample_topics(TopicModelParams tmp, double *topics);

/*
 * Recompute the optimal alpha from the data
 */
void
optimize_alpha(TopicModelParams tmp);

/*
 * Remove all of the documents from the topic model and update the parameters
 * accordingly.
 *
 * This is necessary because inferencing over a new document consists of 
 * adding it to the topic model and then re-sampling it.  So to inference over
 * a new document you would:
 *  1. Make a copy of the model and set the document list to be null
 *  2. Add the document
 *  3. Run a bunch of passes until the new model converges
 * Then you can re-use the original model by running remove_documents().  Or
 * just make a new copy.  It's up to you.
 */
void
remove_documents(TopicModelParams *tmp);

/*
 * Fill doctopics[] with the document-topic matrix so that
 *      doctopics[t*T + d] 
 * is the number of times document d has a token assigned to topic t
 */
void
flat_sample_documents(TopicModelParams tmp, double *doctopics);

#endif

