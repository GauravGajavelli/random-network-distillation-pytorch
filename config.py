import configparser
import os

config = configparser.ConfigParser()
config.read('./config.conf')

# CONFIG_EXTRA: optional second config file whose [DEFAULT] values override the
# primary config.conf. Used by cuda/ run scripts to inject UseAMP, CudnnBenchmark,
# etc. without duplicating every experiment section. Later reads win on conflict.
extra = os.environ.get('CONFIG_EXTRA', '').strip()
if extra:
    config.read(extra)

# ---------------------------------
# Section selection
# Set CONFIG_SECTION=<section> to pick a non-default section.
# Defaults to DEFAULT (legacy Atari config) for backwards compatibility.
# ---------------------------------
default_section = os.environ.get('CONFIG_SECTION', 'DEFAULT').strip()
if default_section != 'DEFAULT' and default_section not in config.sections():
    raise KeyError(
        f"CONFIG_SECTION={default_section!r} not found. "
        f"Available sections: {config.sections()}"
    )
default_config = config[default_section]
