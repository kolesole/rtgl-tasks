"""Collection of pre-defined RTGL tasks on RelBench Amazon dataset."""

import pandas as pd
from relbench import load_dataset
from relbench.base import TaskType

from rtgl_tasks.base import RTGLTmpTask

########### Temporal Tasks ###########

class UserChurnTmpTask(RTGLTmpTask):
    """For each user, predict 1 if the customer does not review any product in the next 3 months, and 0 otherwise."""

    dataset = load_dataset("rel-amazon")
    entity_table = "customer"
    task_type = TaskType.BINARY_CLASSIFICATION

    timedelta = pd.Timedelta(days=365//4)
    val_timestamp = dataset.val_timestamp
    test_timestamp = dataset.test_timestamp

    rtgl_query = """
        PREDICT COUNT(review.*, 0, 91, DAYS) == 0
        FOR EACH customer.*
        WHERE COUNT(review.*, -91, 0, DAYS) != 0;
    """


class ItemChurnTmpTask(RTGLTmpTask):
    """For each product, predict 1 if the product does not receive any reviews in the next 3 months."""

    dataset = load_dataset("rel-amazon")
    entity_table = "product"
    task_type = TaskType.BINARY_CLASSIFICATION

    timedelta = pd.Timedelta(days=365//4)
    val_timestamp = dataset.val_timestamp
    test_timestamp = dataset.test_timestamp

    rtgl_query = """
        PREDICT COUNT(review.*, 0, 91, DAYS) == 0
        FOR EACH product.*
        WHERE COUNT(review.*, -91, 0, DAYS) != 0;
    """


class UserLTVTmpTask(RTGLTmpTask):
    """For each user, predict the $ value of the total number of products they buy and review in the next 3 months."""

    dataset = load_dataset("rel-amazon")
    entity_table = "customer"
    task_type = TaskType.REGRESSION

    timedelta = pd.Timedelta(days=365//4)
    val_timestamp = dataset.val_timestamp
    test_timestamp = dataset.test_timestamp

    rtgl_query = """
        PREDICT SUM(product.price, 0, 91, DAYS)
        FOR EACH customer.*
        WHERE COUNT(review.*, -91, 0, DAYS) != 0;
    """


class UserItemPurchaseTmpTask(RTGLTmpTask):
    """Predict the list of distinct items each customer will purchase in the next 3 months."""

    dataset = load_dataset("rel-amazon")
    entity_table = "customer"
    task_type = TaskType.RECOMMENDATION
    dst_table = "product"

    timedelta = pd.Timedelta(days=365//4)
    val_timestamp = dataset.val_timestamp
    test_timestamp = dataset.test_timestamp

    rtgl_query = """
        PREDICT LIST_DISTINCT(product.*, 0, 91, DAYS)
        FOR EACH customer.*;
    """


class UserItemRateTmpTask(RTGLTmpTask):
    """Predict the list of distinct items each customer will purchase and give a 5 star review in the next 3 months."""

    dataset = load_dataset("rel-amazon")
    entity_table = "customer"
    task_type = TaskType.RECOMMENDATION
    dst_table = "product"

    timedelta = pd.Timedelta(days=365//4)
    val_timestamp = dataset.val_timestamp
    test_timestamp = dataset.test_timestamp

    rtgl_query = """
        PREDICT LIST_DISTINCT(review.product_id WHERE review.rating == 5, 0, 91, DAYS)
        FOR EACH customer.*;
    """


class ItemLTVTmpTask(RTGLTmpTask):
    """For each product, predict the $ value of total number purchases and reviews it recieves in the next 3 months."""

    dataset = load_dataset("rel-amazon")
    entity_table = "product"
    task_type = TaskType.REGRESSION

    timedelta = pd.Timedelta(days=365//4)
    val_timestamp = dataset.val_timestamp
    test_timestamp = dataset.test_timestamp

    rtgl_query = """
        WITH product_product AS (
            product.product_id->review.product_id->product.product_id
        )
        PREDICT SUM(product_product.price, 0, 91, DAYS)
        FOR EACH product.*
        ASSUMING COUNT(review.*, 0, 91, DAYS) != 0;
    """


class ItemLTVTmpTaskInjection(RTGLTmpTask):
    """For each product, predict the $ value of total number purchases and reviews it recieves in the next 3 months."""

    dataset = load_dataset("rel-amazon")
    entity_table = "product"
    task_type = TaskType.REGRESSION

    timedelta = pd.Timedelta(days=365//4)
    val_timestamp = dataset.val_timestamp
    test_timestamp = dataset.test_timestamp

    rtgl_query = """
        PREDICT SUM(
            [
                SELECT
                    p.product_id,
                    p.price,
                    r.review_time
                FROM
                    product AS p
                JOIN
                    review AS r
                ON
                    r.product_id = p.product_id
            ]{product_product}
            {}
            {product_id->product}
            {}
            {review_time}.price, 0, 91, DAYS)
        FOR EACH product.*
        ASSUMING COUNT(review.*, 0, 91, DAYS) != 0;
    """


class UserItemReviewTmpTaskInjection(RTGLTmpTask):
    """Predict the list of distinct items each customer will purchase and give a detailed review in the next 3 months."""

    dataset = load_dataset("rel-amazon")
    entity_table = "customer"
    task_type = TaskType.RECOMMENDATION
    dst_table = "product"

    timedelta = pd.Timedelta(days=365//4)
    val_timestamp = dataset.val_timestamp
    test_timestamp = dataset.test_timestamp

    rtgl_query = """
        PREDICT LIST_DISTINCT(
            [
                SELECT
                    *
                FROM
                    review
                WHERE
                    LENGTH(review_text) > 300
                    AND review_text IS NOT NULL
            ]{filtered_review}
            {}
            {product_id->product, customer_id->customer}
            {}
            {review_time}.product_id, 0, 91, DAYS)
        FOR EACH customer.*;
    """
