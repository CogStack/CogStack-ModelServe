import asyncio
from functools import partial, wraps
from pytest_bdd import parsers


def parse_data_table(text, orient="dict"):
    parsed_text = [
        [x.strip() for x in line.split("|")]
        for line in [x.strip("|") for x in text.splitlines()]
    ]

    header, *data = parsed_text

    if orient == "dict":
        return [
            dict(zip(header, line))
            for line in data
        ]
    else:
        if orient == "columns":
            data = [
                [line[i] for line in data]
                for i in range(len(header))
            ]
        return header, data


def data_table(name, fixture="data", orient="dict"):
    formatted_str = "{name}\n{{{fixture}:DataTable}}".format(
        name=name,
        fixture=fixture,
    )
    data_table_parser = partial(parse_data_table, orient=orient)

    return parsers.cfparse(formatted_str, extra_types=dict(DataTable=data_table_parser))


def async_to_sync(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return wrapper
