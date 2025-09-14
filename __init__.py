"""The Gemini Live integration."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.typing import ConfigType
from homeassistant.core import HomeAssistant
from homeassistant.const import CONF_API_KEY, Platform
from dotenv import load_dotenv


from .core.app import GeminiApp

DOMAIN = "gemini_live_hass_integration"

SATELLITE_PLATFORMS = [
    Platform.SWITCH,
    Platform.BINARY_SENSOR,
    Platform.SENSOR,
]

async def main(hass: HomeAssistant):
    load_dotenv()
    app = GeminiApp(hass=hass)
    await app.run()

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Load Gemini STT."""

    # await hass.config_entries.async_forward_entry_setups(entry, SATELLITE_PLATFORMS)




    config_entry.async_on_unload(config_entry.add_update_listener(update_listener))

    config_entry.async_create_background_task(
        hass,
        main(hass),
        f"Starting Gemini Application...",
    )

    return True


async def update_listener(hass: HomeAssistant, entry: ConfigEntry):
    """Handle options update."""
    await hass.config_entries.async_reload(entry.entry_id)


# async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
#     """Unload a config entry."""

#     return await hass.config_entries.async_unload_platforms(entry, SATELLITE_PLATFORMS)


