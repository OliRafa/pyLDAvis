"""
pyLDAvis Tomotopy
===============
Helper functions to visualize Tomotopy's LatentDirichletAllocation models
"""

from typing import Any, Dict, List, Optional

import funcy as fp
import numpy as np

import pyLDAvis


def _get_document_lengths(model) -> np.ndarray:
    return np.array([len(document.words) for document in model.docs])


def _get_term_frequencies(model) -> List:
    return model.used_vocab_freq


def _get_vocabulary(model) -> List:
    return list(model.used_vocabs)


def _normalize_values(values: np.ndarray) -> np.ndarray:
    return values / values.sum(axis=1, keepdims=True)


def _get_document_topic_distributions(model, live_topics: Optional[List[int]] = None):
    document_topic_distributions = np.stack(
        [document.get_topic_dist() for document in model.docs]
    )

    if live_topics:
        document_topic_distributions = document_topic_distributions[:, live_topics]

    return _normalize_values(document_topic_distributions)


def _get_topic_term_distributions(
    model, live_topics: Optional[List[int]] = None
) -> np.ndarray:
    topic_term_distributions = np.stack(
        [model.get_topic_word_dist(topic_id) for topic_id in range(model.k)]
    )

    if live_topics:
        topic_term_distributions = topic_term_distributions[live_topics]

    return _normalize_values(topic_term_distributions)


def _get_live_topics(model) -> List[int]:
    return [topic_id for topic_id in range(model.k) if model.is_live_topic(topic_id)]


def _extract_data(model, live_topics_only: bool = True) -> Dict[str, Any]:
    if live_topics_only:
        live_topics = _get_live_topics(model)

    else:
        live_topics = None

    vocabulary = _get_vocabulary(model)
    document_lengths = _get_document_lengths(model, live_topics)
    term_frequencies = _get_term_frequencies(model, live_topics)
    topic_term_distributions = _get_topic_term_distributions(model, live_topics)

    error_message = (
        "Term frequencies and vocabulary are of different sizes, "
        f"{term_frequencies.shape[0]} != {len(vocabulary)}."
    )
    assert term_frequencies.shape[0] == len(vocabulary), error_message

    error_message = (
        "Topic-term distributions and document-term matrix have different number "
        f"of columns, {topic_term_distributions.shape[1]} != {len(vocabulary)}."
    )
    # assert topic_term_distributions.shape[1] == dtm.shape[1], error_message

    # column dimensions of document-term matrix and topic-term distributions
    # must match first before transforming to document-topic distributions
    document_topic_distributions = _get_document_topic_distributions(model, live_topics)

    assert topic_term_distributions.shape[0] == document_topic_distributions.shape[1]

    return {
        "vocab": vocabulary,
        "doc_lengths": document_lengths,
        "term_frequency": term_frequencies,
        "doc_topic_dists": document_topic_distributions,
        "topic_term_dists": topic_term_distributions,
        # Tomotopy starts topic ids with 0 and pyLDAvis with 1.
        "start_index": 0,
        # Otherwise the topic_ids between pyLDAvis and Tomotopy don't match.
        "sort_topics": False,
    }


def prepare(model, live_topics_only: bool = True, **kwargs):
    """Create Prepared Data from Tomotopy's Latent Dirichlet Allocation.

    Parameters
    ----------
    model : tomotopy.LDAModel.
        Latent Dirichlet Allocation model from Tomotopy.

    live_topics_only : bool
        Only consider live topics for the visualization.

    **kwargs : Dict
        Keyword argument to be passed to pyLDAvis.prepare().

    Returns
    -------
    prepared_data : PreparedData
          The data structures used in the visualization.


    Example
    --------
    For example usage please see this notebook:
    http://nbviewer.ipython.org/github/bmabey/pyLDAvis/blob/master/notebooks/sklearn.ipynb

    See
    ---
    See `pyLDAvis.prepare` for **kwargs.

    """
    opts = fp.merge(_extract_data(model, live_topics_only), kwargs)
    return pyLDAvis.prepare(**opts)
