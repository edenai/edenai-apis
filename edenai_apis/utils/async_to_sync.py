from time import sleep
import asyncio
from typing import Callable


def fibonacci_waiting_call(
    max_time: int,
    status: str,
    func: Callable,
    status_positif: bool = True,
    provider_handel_call: Callable = None,
    **func_args,
):
    """Check response call if succeeded synchronously form an async endpoint

    Args:
        max_time (int): Max time to wait
        status (str): The success/wating string to check with if job finished
        func (Callable): The function to call each time to check if job finished
        status_positif (int): Wether to test status with a sucess or a waiting one
        provider_handel_call (Callable): The function wrapper for the provider call
        to handle errors
    """
    first_occurence, second_occurence = (
        1,
        2,
    )  # waiting exponentially using fibonacci
    wait_time = first_occurence + second_occurence
    total_wait_time = wait_time
    get_response = (
        func(**func_args)
        if not provider_handel_call
        else provider_handel_call(func, **func_args)
    )
    while total_wait_time < max_time:  # Wait for the answer from provider
        if (status_positif and get_response["JobStatus"] == status) or (
            not status_positif and get_response["JobStatus"] != status
        ):
            break
        sleep(wait_time)
        first_occurence = second_occurence
        second_occurence = wait_time
        wait_time = first_occurence + second_occurence
        total_wait_time += wait_time
        if provider_handel_call:
            get_response = provider_handel_call(func, **func_args)
        else:
            get_response = func(**func_args)
    return get_response


async def afibonacci_waiting_call(
    max_time: int,
    status: str,
    func: Callable,
    status_positif: bool = True,
    provider_handel_call: Callable = None,
    **func_args,
):
    """Check response call if succeeded asynchronously from an async endpoint
    Args:
        max_time (int): Max time to wait
        status (str): The success/waiting string to check with if job finished
        func (Callable): The async function to call each time to check if job finished
        status_positif (int): Whether to test status with a success or a waiting one
        provider_handel_call (Callable): The async function wrapper for the provider call
        to handle errors
    """
    first_occurence, second_occurence = (
        1,
        2,
    )  # waiting exponentially using fibonacci
    wait_time = first_occurence + second_occurence
    total_wait_time = wait_time

    # Initial call
    get_response = (
        await func(**func_args)
        if not provider_handel_call
        else await provider_handel_call(func, **func_args)
    )

    while total_wait_time < max_time:  # Wait for the answer from provider
        if (status_positif and get_response["JobStatus"] == status) or (
            not status_positif and get_response["JobStatus"] != status
        ):
            break

        await asyncio.sleep(wait_time)

        first_occurence = second_occurence
        second_occurence = wait_time
        wait_time = first_occurence + second_occurence
        total_wait_time += wait_time

        if provider_handel_call:
            get_response = await provider_handel_call(func, **func_args)
        else:
            get_response = await func(**func_args)

    return get_response
