from dip_course.cli.official_split import parse_official_ids


def test_parse_official_ids() -> None:
    content = b"./Canon/0.0001/1.raw ./Canon/gt/1.jpg\n./Canon/0.0001/9.raw ./Canon/gt/9.jpg\n"
    assert parse_official_ids(content) == {"1", "9"}
