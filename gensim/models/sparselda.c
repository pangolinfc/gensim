#include <stdlib.h>
#include <string.h>

#include "sparselda.h"

unsigned int
compute_topic_mask(unsigned int T)
{
    unsigned int mask = 0;
    unsigned int i = T-1;
    while(i)
    {
        mask |= i;
        i = i >> 1;
    }
    return mask;
}

unsigned int
compute_topic_bits(unsigned int T)
{
    unsigned int bits = 0;
    unsigned int i = T-1;
    while(i)
    {
        ++bits;
        i = i >> 1;
    }
    return bits;
}

void
free_feature_sequence(FeatureSequence *fs)
{
    free(fs->Ntd);
    free(fs->topics);
    free(fs->tokens);
}

void
free_feature_sequence_list(FeatureSequence *fs)
{
    free_feature_sequence(fs);
    if (fs->next)
    {
        free_feature_sequence_list(fs->next);
        // free(fs->next);
    }
    free(fs);
}

void
free_topic_model_params(TopicModelParams *tmp)
{
    free(tmp->q_);
    free(tmp->denoms);

    if(tmp->Nwt)
    {
        for (int i=0; i<tmp->V; i++)
        {
            free(tmp->Nwt[i]);
        }
    }
    free(tmp->Nwt);
    free(tmp->Nt);
    free(tmp->alpha);

    // Free the histograms
    free(tmp->Cn);
    if(tmp->Ctn)
    {
        for (int i=0; i<tmp->T; i++)
        {
            free(tmp->Ctn[i]);
        }
    }
    free(tmp->Ctn);
    if (tmp->documents)
    {
        free_feature_sequence_list(tmp->documents);
    }
}

int
new_topic_model_params(TopicModelParams *tmp, unsigned int T)
{
    // Clear all of the pointers to NULL
    memset(tmp, 0, sizeof(TopicModelParams));

    unsigned int t;
    tmp->T = T;
    tmp->V = 0;
    tmp->N = 0;
    tmp->Tmask = compute_topic_mask(T);
    tmp->Tbits = compute_topic_bits(T);
    tmp->D = 0;
    tmp->beta = 0.01;
    tmp->documents = NULL;

    /*
     * Allocating memory
     */
    tmp->alpha = (double *)malloc(sizeof(double)*T);
    if (!tmp->alpha) goto fail;

    tmp->Nt = (unsigned int *)calloc(T, sizeof(unsigned int));
    if (!tmp->Nt) goto fail;

    tmp->denoms = (double *)malloc(sizeof(double)*T);
    if (!tmp->denoms) goto fail;

    tmp->q_ = (double *)malloc(sizeof(double)*T);
    if (!tmp->q_) goto fail;

    /*
     * Initializing parameters
     */
    for (t=0; t<T; t++)
    {
        tmp->alpha[t] = 50.0/T;
    }

    return 1;
fail:
    free_topic_model_params(tmp);
    return 0;
}

int
new_inferencer(TopicModelParams *inferencer, TopicModelParams *tmp)
{
    // Clear all of the pointers to NULL
    memset(inferencer, 0, sizeof(TopicModelParams));

    unsigned int t;
    inferencer->T = tmp->T;
    inferencer->V = tmp->V;
    inferencer->N = tmp->N;
    inferencer->Tmask = tmp->Tmask;
    inferencer->Tbits = tmp->Tbits;
    inferencer->D = 0;
    inferencer->beta = tmp->beta;

    /*
     * Allocating memory
     */
    inferencer->alpha = (double *)malloc(sizeof(double)*tmp->T);
    if (!inferencer->alpha) goto fail;

    inferencer->Nt = (unsigned int *)calloc(tmp->T, sizeof(unsigned int));
    if (!inferencer->Nt) goto fail;

    inferencer->denoms = (double *)malloc(sizeof(double)*tmp->T);
    if (!inferencer->denoms) goto fail;

    inferencer->q_ = (double *)malloc(sizeof(double)*tmp->T);
    if (!inferencer->q_) goto fail;

    /*
     * Initializing parameters
     */
    memcpy(inferencer->alpha, tmp->alpha, sizeof(double)*inferencer->T);
    memcpy(inferencer->Nt, tmp->Nt, sizeof(unsigned int)*inferencer->T);

    return 1;
fail:
    free_topic_model_params(tmp);
    return 0;
}

int
init_feature_sequence(
        TopicModelParams *tmp, FeatureSequence *fs, unsigned int length)
{
    memset(fs, 0, sizeof(FeatureSequence));

    fs->length = length;
    fs->tokens = (unsigned int *)malloc(sizeof(unsigned int)*length);
    if (!fs->tokens) goto fail;
    fs->topics = (unsigned int *)malloc(sizeof(unsigned int)*length);
    if (!fs->topics) goto fail;
    fs->Ntd = (unsigned int *)calloc(tmp->T, sizeof(unsigned int));
    if (!fs->Ntd) goto fail;

    return 1;
fail:
    free_feature_sequence(fs);
    return 0;
}

/*
 * Bubble sort the tuple in the specified index up
 */
void
bubble_up(unsigned int index, unsigned int *wt)
{
    unsigned int swapper;
    while(index > 0 && wt[index] > wt[index-1])
    {
        swapper = wt[index];
        wt[index] = wt[index-1];
        wt[index-1] = swapper;
        --index;
    }
}

/*
 * Bubble sort the tuple in the specified index down
 */
void
bubble_down(unsigned int index, unsigned int *wt, unsigned int T)
{
    unsigned int swapper;
    while(index < T-1 && wt[index] < wt[index+1])
    {
        swapper = wt[index];
        wt[index] = wt[index+1];
        wt[index+1] = swapper;
        ++index;
    }
}

/*
 * Find the specified word/topic in the word/topic matrix, increment the
 * count, and bubble sort it up.
 */
void
find_and_increment(unsigned int t, unsigned int w, TopicModelParams tmp)
{
    unsigned int i = 0;
    while (i < tmp.T && ((tmp.Nwt[w][i] & tmp.Tmask) != t))
    {
        ++i;
    }
    
    if (i == tmp.T)
    {
        /*
         * We haven't seen this word assigned to this topic.  Stick it in the
         * last slot and bubble it up.
         */
        i = tmp.T-1;
        tmp.Nwt[w][i] = (1 << tmp.Tbits) | t;
    }
    else
    {
        tmp.Nwt[w][i] += 1 << tmp.Tbits;
    }
    bubble_up(i, tmp.Nwt[w]);
}

/*
 * Find the specified word/topic in the word/topic matrix, decrement the
 * count, and bubble sort it down.
 */
void
find_and_decrement(unsigned int t, unsigned int w, TopicModelParams tmp)
{
    unsigned int i = 0;
    while (i < tmp.T && ((tmp.Nwt[w][i] & tmp.Tmask) != t))
    {
        ++i;
    }
    
    if (i == tmp.T)
    {
        /*
         * This should never, ever happen.  I should probably stick in a 
         * `require' to cover this case.
         */
    }
    else
    {
        tmp.Nwt[w][i] -= 1 << tmp.Tbits;
        bubble_down(i, tmp.Nwt[w], tmp.T);
    }
}

int add_document(
        TopicModelParams *tmp, unsigned int *tokens, unsigned int length)
{
    FeatureSequence *fs = (FeatureSequence *)malloc(sizeof(FeatureSequence));
    if(!fs) goto fail;
    if(!init_feature_sequence(tmp, fs, length)) goto fail;

    if (length > tmp->max_len)
    {
        tmp->max_len = length;
    }

    /*
     * Initialize the document to random topics and update all of the stats.
     */
    for(int i=0; i<length; i++)
    {
        int t = rand() % tmp->T;
        fs->tokens[i] = tokens[i];
        if (tokens[i] >= tmp->V)
        {
            tmp->V = tokens[i]+1;
        }
        tmp->Nt[t]++;
        fs->Ntd[t]++;
        fs->topics[i] = t;
    }
    fs->next = tmp->documents;
    tmp->documents = fs;

    tmp->D++;
    tmp->D += length;

    return 1;
fail:
    free_feature_sequence(fs);
    free(fs);
    return 0;
}

void
remove_documents(TopicModelParams *tmp)
{
    FeatureSequence *fs;

    /*
     * Fix all of the stats
     */
    for (fs=tmp->documents; fs!=NULL; fs=fs->next)
    {
        for (int t=0; t<tmp->T; t++)
        {
            tmp->Nt[t] -= fs->Ntd[t];
        }
        for (int i=0; i<fs->length; i++)
        {
            find_and_decrement(fs->topics[i], fs->tokens[i], *tmp);
        }
    }
    free_feature_sequence_list(tmp->documents);
    tmp->documents = NULL;

    /*
     * Clear out the histogram -- we don't need it any more.
     */
    free(tmp->Cn);
    tmp->Cn = NULL;
    for (int t=0; t<tmp->T; t++)
    {
        free(tmp->Ctn[t]);
    }
    free(tmp->Ctn);
    tmp->Ctn = NULL;
}

void
do_document(FeatureSequence fs, TopicModelParams tmp)
{

    // Pre-compute everything we possibly can.
    double s = 0.0;
    double r = 0.0;
    for (int t=0; t<tmp.T; ++t)
    {
        tmp.denoms[t] = 1.0/(tmp.beta*tmp.V + tmp.Nt[t]);
        s += tmp.alpha[t]*tmp.beta*tmp.denoms[t];
        r += fs.Ntd[t]*tmp.beta*tmp.denoms[t];
        tmp.q_[t] = (tmp.alpha[t] + fs.Ntd[t])*tmp.denoms[t];
        --tmp.Ctn[t][fs.Ntd[t]];
    }

    for (int i=0; i<fs.length; ++i)
    {
        /*
         * Removing the current word from its old topic...
         */
        unsigned int w = fs.tokens[i];
        unsigned int z = fs.topics[i];

        s -= tmp.alpha[z]*tmp.beta*tmp.denoms[z];
        r -= tmp.beta*fs.Ntd[z]*tmp.denoms[z];

        --(fs.Ntd[z]);
        --(tmp.Nt[z]);
        tmp.denoms[z] = 1.0/(tmp.beta*tmp.V+tmp.Nt[z]);

        s += tmp.alpha[z]*tmp.beta*tmp.denoms[z];
        r += tmp.beta*fs.Ntd[z]*tmp.denoms[z];
        tmp.q_[z] = (tmp.alpha[z] + fs.Ntd[z])*tmp.denoms[z];

        unsigned int j=0;
        for (j=0; j<tmp.T; ++j)
        {
            if ((tmp.Nwt[w][j] & tmp.Tmask) == z 
                    && (tmp.Nwt[w][j] >> tmp.Tbits > 0))
            {
                tmp.Nwt[w][j] -= (1 << tmp.Tbits);
                break;
            }
        }
        bubble_down(j, tmp.Nwt[w], tmp.T);

        double q = 0.0;
        for (int j=0; j<tmp.T; ++j)
        {
            if ((tmp.Nwt[w][j] >> tmp.Tbits) == 0)
            {
                break;
            }
            q += tmp.q_[tmp.Nwt[w][j] & tmp.Tmask]*(tmp.Nwt[w][j] >> tmp.Tbits);
        }

        // Do our random draw from the multinomial
        double x = (s+r+q)*((double)rand()/RAND_MAX);

        unsigned int new_z;
        if (x < s)
        {
            // Smoothing case -- draw a random topic from the prior
            
            // x /= tmp.alpha[z]*tmp.beta;
            for (new_z=0; new_z<tmp.T; ++new_z)
            {
                x -= tmp.alpha[new_z]*tmp.beta*tmp.denoms[new_z];
                if (x <= 0) 
                {
                    find_and_increment(new_z, w, tmp);
                    break;
                }
            }
        }
        else if (x < s+r)
        {
            // Smoothing case -- draw a random topic according to document
            // topics

            x -= s;
            for (new_z=0; new_z<tmp.T; ++new_z)
            {
                x -= fs.Ntd[new_z]*tmp.beta*tmp.denoms[new_z];
                if (x <= 0)
                { 
                    find_and_increment(new_z, w, tmp);
                    break;
                }
            }
        }
        else
        {
            // Draw from this word's topic distribution

            x -= s + r;
            for (int j=0; j<tmp.T; ++j)
            {
                unsigned int nt = tmp.Nwt[w][j];
                x -= tmp.q_[nt & tmp.Tmask]*(nt >> tmp.Tbits);
                if (x <= 0)
                {
                    new_z = nt & tmp.Tmask;
                    tmp.Nwt[w][j] += 1 << tmp.Tbits;
                    bubble_up(j, tmp.Nwt[w]);
                    break;
                }
            }
        }
        /*
        if (x > 0)
        {
            cout << "PANIC! x = " << x << endl;
            for (int t=0; t<tmp.T; ++t)
            {
                cout << t << ":" << tmp.alpha[t] << " ";
            }
            cout << endl;
        }
        */
        
        /*
         * Update the stats
         */
        s -= tmp.alpha[new_z]*tmp.beta*tmp.denoms[new_z];
        r -= tmp.beta*fs.Ntd[new_z]*tmp.denoms[new_z];

        ++(fs.Ntd[new_z]);
        ++(tmp.Nt[new_z]);
        tmp.denoms[new_z] = 1.0/(tmp.beta*tmp.V+tmp.Nt[new_z]);

        s += tmp.alpha[new_z]*tmp.beta*tmp.denoms[new_z];
        r += tmp.beta*fs.Ntd[new_z]*tmp.denoms[new_z];
        tmp.q_[new_z] = (tmp.alpha[new_z] + fs.Ntd[new_z])*tmp.denoms[new_z];

        fs.topics[i] = new_z;
    }

    for (unsigned int t=0; t<tmp.T; ++t)
    {
        ++tmp.Ctn[t][fs.Ntd[t]];
    }
}

/*
 * Allocate memory for the TopicModelParams
 */
int
prep_topic_model_params(TopicModelParams *tmp)
{
    tmp->Nwt = (unsigned int **)calloc(tmp->V, sizeof(unsigned int *));
    if (!tmp->Nwt) goto fail;
    for(int i=0; i<tmp->V; i++)
    {
        tmp->Nwt[i] = (unsigned int *)calloc(tmp->T, sizeof(unsigned int));
        if (!tmp->Nwt[i]) goto fail;
    }

    tmp->Cn = (unsigned int *)calloc(tmp->max_len+1, sizeof(unsigned int));
    if (!tmp->Cn) goto fail;
    tmp->Ctn = (unsigned int **)calloc(tmp->T, sizeof(unsigned int *));
    if (!tmp->Ctn) goto fail;
    for (int i=0; i<tmp->T; i++)
    {
        tmp->Ctn[i] = (unsigned int *)calloc(tmp->max_len+1, sizeof(unsigned int));
        if (!tmp->Ctn[i]) goto fail;
    }

    for(FeatureSequence *fs=tmp->documents; fs!=NULL; fs=fs->next)
    {
        tmp->Cn[fs->length]++;
        for (int t=0; t<tmp->T; t++)
        {
            tmp->Ctn[t][fs->Ntd[t]]++;
        }
        for (int i=0; i<fs->length; i++)
        {
            find_and_increment(fs->topics[i], fs->tokens[i], *tmp);
        }
    }

    return 1;
fail:
    return 0;
}

void
do_iteration(TopicModelParams *tmp)
{
    if (!tmp->Nwt)
    {
        prep_topic_model_params(tmp);
    }
    for(FeatureSequence *fs=tmp->documents; fs!=NULL; fs=fs->next)
    {
        do_document(*fs, *tmp);
    } 
}

void
sample_topics(TopicModelParams tmp, double **topics)
{
    for (int w=0; w<tmp.V; ++w)
    {
        for (int i=0; i<tmp.T; ++i)
        {
            if ((tmp.Nwt[w][i] >> tmp.Tbits) == 0)
            {
                break;
            }
            unsigned int t = tmp.Nwt[w][i] & tmp.Tmask;
            topics[t][w] += (tmp.Nwt[w][i] >> tmp.Tbits)/(double)tmp.Nt[t];
        }
    }
}

void
flat_sample_topics(TopicModelParams tmp, double *topics)
{
    for (int w=0; w<tmp.V; ++w)
    {
        for (int i=0; i<tmp.T; ++i)
        {
            if ((tmp.Nwt[w][i] >> tmp.Tbits) == 0)
            {
                break;
            }
            unsigned int t = tmp.Nwt[w][i] & tmp.Tmask;
            topics[t*tmp.V+w] += (tmp.Nwt[w][i] >> tmp.Tbits)/(double)tmp.Nt[t];
        }
    }
}

void
flat_sample_documents(TopicModelParams tmp, double *doctopics)
{
    int d=0;
    for (FeatureSequence *fs=tmp.documents; fs!=NULL; fs=fs->next)
    {
        memcpy(doctopics+(d*tmp.T), fs->Ntd, tmp.T*sizeof(double));
    }
}

/*
 * Find the optimal Dirichlet prior for a given collection of multinomial
 * distributions, using a fixed point method and the stats gathered during
 * the iteration.
 */
void
optimize_alpha(TopicModelParams tmp)
{
    unsigned int max_ctn[tmp.T];
    double oldalpha[tmp.T];

    for (unsigned int t=0; t<tmp.T; ++t)
    {
        for (unsigned int n=0; n<=tmp.max_len; ++n)
        {
            if (tmp.Ctn[t][n] > 0)
            {
                max_ctn[t] = n;
            }
        }
    }

    double epsilon = 0.0001;
    double diff = epsilon + 1.0;
    while (diff > epsilon)
    {
        diff = 0.0;
        double sumalpha = 0.0;
        for (unsigned int t=0; t<tmp.T; ++t)
        {
            sumalpha += tmp.alpha[t];
            oldalpha[t] = tmp.alpha[t];
        }

        double S = 0.0;
        double D = 0.0;
        for (unsigned int n=0; n<=tmp.max_len; ++n)
        {
            D += 1.0/(n-1+sumalpha);
            S += tmp.Cn[n]*D;
        }

        for (unsigned int t=0; t<tmp.T; ++t)
        {
            D = 0.0;
            double Sk = 0.0;
            for (unsigned int n=0; n<=max_ctn[t]; ++n)
            {
                D += 1.0/(n-1+tmp.alpha[t]);
                Sk += tmp.Ctn[t][n]*D;
            }

            tmp.alpha[t] *= Sk/S;
            diff += abs(tmp.alpha[t] - oldalpha[t]);
        }
    }
}

