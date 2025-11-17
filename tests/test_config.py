import logging
import warnings
from pathlib import Path

import pytest
import xarray as xr

import xscen as xs
from xscen import CONFIG


ROOT = Path(__file__).parent.parent
CONFIG_FILE1 = str(ROOT / "templates/1-basic_workflow_with_config/config1.yml")
CONFIG_FILE2 = str(ROOT / "templates/2-indicators_only/config2.yml")
CONFIG_DIR1 = str(ROOT / "templates/1-basic_workflow_with_config/")

LOGGER = logging.getLogger(__name__)


def test_log(caplog):
    with caplog.at_level(logging.INFO):
        xs.load_config("test=test4", verbose=True, reset=True)
    assert "Updated the config with test=test4" in caplog.text


def test_load_config():
    xs.load_config(CONFIG_FILE1)
    # test basic assignment
    assert CONFIG["scripting"]["send_mail_on_exit"]["subject"] == "Template 1 - basic_workflow_with_config"
    # test external
    assert xr.get_options()["display_style"] == "text"
    assert warnings.filters[0][0] == "always"
    assert warnings.filters[0][2] is Warning

    # adding new config file
    xs.load_config(CONFIG_FILE2)
    # existing keys are updated
    assert CONFIG["scripting"]["send_mail_on_exit"]["subject"] == "Indicator computing terminated."
    # old keys that are not in the new config remain
    assert CONFIG["project"]["name"] == "Template 1 - basic_workflow_with_config"
    # test external
    assert warnings.filters[0][0] == "ignore"
    assert warnings.filters[0][2] is FutureWarning

    # reset config
    xs.load_config(CONFIG_FILE2, reset=True)
    # after reset old keys are gone
    assert "project" not in CONFIG

    # loading from key=value
    xs.load_config("test=test1")
    assert CONFIG["test"] == "test1"

    # loading from dir
    xs.load_config(CONFIG_DIR1, verbose=True)
    assert CONFIG["scripting"]["send_mail_on_exit"]["subject"] == "Template 1 - basic_workflow_with_config"


def test_set_config():
    CONFIG.set("test", "test2")

    assert CONFIG["test"] == "test2"

    CONFIG.set("regrid.regrid_datasets.to_level", "reg")

    assert CONFIG["regrid"]["regrid_datasets"]["to_level"] == "reg"

    # not an existing module, can't use "."
    with pytest.raises(ValueError):
        CONFIG.set("test.subtest", "test2")


def test_update_config():
    CONFIG.update_from_list([("test", "test3")])

    assert CONFIG["test"] == "test3"


def test_get_configurable():
    d = xs.config.get_configurable()
    assert len(d) == 32  # number of @parse_config

    CONFIG.clear()  # reset config after tests
