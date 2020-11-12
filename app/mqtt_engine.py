import json
import logging

import paho.mqtt.client as mqtt

class MQTTEngine:
    '''
    MQTTEngine is a class for MQTT communication.
    @param cfg: cfg['mqtt_engine']
    '''
    def __init__(self, cfg):
        # Logging
        self.init_logger()

        # Variables
        self.broker_ip = cfg['broker_ip']
        self.broker_port = cfg['broker_port']
        self.pub_topic = cfg['pub_topic']

        # MQTT Client
        self.client = mqtt.Client()

    def init_logger(self):
        '''
        Initiate a logger for MainEngine.
        '''
        logger = logging.getLogger('Main.MQTTEngine')
        logger.setLevel(logging.INFO)
        self.logger = logger

    def connect(self):
        '''
        Connect to MQTT Broker.
        '''
        self.client.connect_async(self.broker_ip, self.broker_port)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.loop_start()

    def close(self):
        '''
        Close the MQTT connection.
        '''
        self.client.loop_stop()
        self.client.disconnect()

    def publish(self, body):
        '''
        Publish message as the specific topic.
        '''
        self.client.publish(self.pub_topic, json.dumps(body), 1)

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.logger.info('made connection with MQTT broker successfully.')
        else:
            self.logger.info('MQTT connection failed. Code =' + str(rc))

    def _on_disconnect(self, client, userdata, flags, rc=0):
        self.logger.info('MQTT connection disconnected.')
