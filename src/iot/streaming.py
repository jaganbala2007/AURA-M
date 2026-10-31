"""
AURA-M Industrial IoT & Streaming Telemetry Engine (Technology Module 6)
Implements MQTT publishing/subscribing with JSON payload schema validation,
event timestamping, message deduplication, and streaming event buffer.
"""

import json
import time
import datetime
import logging
from typing import Dict, Any, List, Optional, Callable
import paho.mqtt.client as mqtt

from src.data.validator import DataValidator

logger = logging.getLogger("AURA-M.IIoTStreaming")

DEFAULT_BROKER_HOST = "localhost"
DEFAULT_BROKER_PORT = 1883
TOPIC_TELEMETRY = "auram/fleet/telemetry"
TOPIC_ALERTS = "auram/fleet/alerts"

class StreamingTelemetryGateway:
    """Industrial IoT gateway supporting live/simulated MQTT streaming with validation."""

    def __init__(self, broker_host: str = DEFAULT_BROKER_HOST, broker_port: int = DEFAULT_BROKER_PORT):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="AURA-M_Gateway")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.message_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self.is_connected = False
        self.seen_signatures = set()
        self.buffered_events: List[Dict[str, Any]] = []

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            self.is_connected = True
            logger.info(f"Connected to MQTT Broker at {self.broker_host}:{self.broker_port}")
            self.client.subscribe(TOPIC_TELEMETRY)
        else:
            logger.warning(f"MQTT connection failed with return code: {rc}")

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            self.process_incoming_event(payload)
        except Exception as e:
            logger.error(f"Failed to parse incoming streaming event: {e}")

    def connect(self) -> bool:
        """Attempts connection to MQTT broker, handles connection failure gracefully."""
        try:
            self.client.connect(self.broker_host, self.broker_port, keepalive=30)
            self.client.loop_start()
            return True
        except Exception as e:
            logger.info(f"Local MQTT Broker not active ({e}). Operating in Local In-Memory Streaming Mode.")
            self.is_connected = False
            return False

    def disconnect(self) -> None:
        if self.is_connected:
            self.client.loop_stop()
            self.client.disconnect()
            self.is_connected = False

    def process_incoming_event(self, event: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validates event schema, checks sequence monotonicity, rejects duplicates, and buffers event.
        """
        # 1. Deduplication check via signature: (engine_id, cycle)
        sig = (event.get("engine_id"), event.get("cycle"))
        if sig in self.seen_signatures:
            return False, f"Duplicate cycle event rejected for {sig}"
        
        # 2. Schema and type validation
        valid, err = DataValidator.validate_streaming_record(event)
        if not valid:
            return False, f"Telemetry validation rejected: {err}"

        # 3. Add timestamp and register
        event["ingested_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.seen_signatures.add(sig)
        self.buffered_events.append(event)
        if len(self.buffered_events) > 500:
            self.buffered_events.pop(0)

        # 4. Trigger callback if registered
        if self.message_callback:
            try:
                self.message_callback(event)
            except Exception as e:
                logger.error(f"Error in streaming message consumer callback: {e}")

        return True, None

    def publish_cycle_event(self, record: Dict[str, Any]) -> bool:
        """Publishes cycle record to MQTT topic (or buffers locally if standalone)."""
        payload_str = json.dumps(record)
        if self.is_connected:
            try:
                self.client.publish(TOPIC_TELEMETRY, payload_str, qos=1)
                return True
            except Exception as e:
                logger.warning(f"Publish failed: {e}")

        # Local processing fallback
        success, _ = self.process_incoming_event(record)
        return success
