import discord
import argparse
import os
import sys

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('msg_p')
    parser.add_argument('log_p')
    return parser.parse_args()

args = parse_args()

with open('runscripts/dsc.txt', 'r') as dsc_f:
    dsc_token = dsc_f.readline().strip()

client = discord.Client()


@client.event
async def on_ready():
    try:
        with open('runscripts/dsc.txt', 'r') as dsc_f:
            _ = dsc_f.readline().strip()
            channel_id = int(dsc_f.readline().strip())

        with open(args.msg_p, 'r') as msg_f:
            msg = ''.join(msg_f.readlines())

        channel = client.get_channel(channel_id)
        if os.path.exists(args.log_p):
            await channel.send(msg, file=discord.File(args.log_p))
        else:
            await channel.send(msg)
    except:
        sys.exit('Error during message send')

    sys.exit('Message sent.')

client.run(dsc_token)