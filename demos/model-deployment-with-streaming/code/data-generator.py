from random import randint, random
from datetime import datetime, timedelta
import json
import uuid
from faker import Faker
import v3io.dataplane
import os
from mlrun.execution import MLClientCtx


def gen_postcode(is_churn):
    # if is_churn is true the postcode modulu 3 will return 0 or 1
    # if is_churn is false the postcode modulu 3 will return 0 or 2
    # this will encode information in postcode that our ML model will learn
    base_postcode = 3 * randint(3334, 33333)
    group = randint(0, 1)
    if is_churn:
        return base_postcode + group
    else:
        return base_postcode + (group * 2)


# event functions
def new_registration(fake, user_id, event_time, is_churn):
    return {'user_id': user_id,
            'event_type': 'registration',
            'event_time': event_time,
            'name': fake.name(),
            'date_of_birth': fake.date(),
            'street_address': fake.street_address(),
            'city': fake.city(),
            'country': fake.country(),
            'postcode': gen_postcode(is_churn),
            'affiliate_url': fake.image_url(),
            'campaign': fake.ean8()}


def new_purchase(fake, user_id, event_time):
    return {'user_id': user_id,
            'event_type': 'purchase',
            'event_time': event_time,
            'amount': fake.randomize_nb_elements(number=50)}


def new_bet(fake, user_id, event_time):
    return {'user_id': user_id,
            'event_type': 'bet',
            'event_time': event_time,
            'bet_amount': fake.randomize_nb_elements(number=10)}


def new_win(fake, user_id, event_time):
    return {'user_id': user_id,
            'event_type': 'win',
            'event_time': event_time,
            'win_amount': fake.randomize_nb_elements(number=200)}


def gen_event_date(is_churn, prev_event_date=None):
    if prev_event_date is None:
        # generate first event date
        return datetime.now() - timedelta(hours=randint(48, 96))
    else:
        if prev_event_date + timedelta(hours=30) < datetime.now() and not is_churn and randint(1, 1000) <= 5:
            # if the user is not churned and it is possible, generate event in the following day with prbability 0.005
            return prev_event_date + timedelta(hours=randint(15, 24))
        else:
            return prev_event_date + timedelta(seconds=randint(5, 100))


def generate_events(fake, num_users, events_dist, num_events, is_churn):
    user_ids = generate_user_ids(num_users)
    events = []
    for user_id in user_ids:
        # register
        event_time = gen_event_date(is_churn)
        reg_event = new_registration(fake, user_id, event_time, is_churn)
        reg_event['label'] = int(is_churn)
        events.append(reg_event)
        for _ in range(num_events):
            # generate event according to dist
            acc_prob = 0
            rand = random()
            for event_dist in events_dist:
                if rand <= event_dist['probability'] + acc_prob:
                    event_time = gen_event_date(is_churn, event_time)
                    new_event = event_dist['generator'](fake, user_id, event_time)
                    events.append(new_event)
                    break
                else:
                    acc_prob += event_dist['probability']
    return events


def generate_user_ids(n: int):
    return (str(uuid.uuid4()) for _ in range(n))


def generate_event_stream(v3io_client,
                          container,
                          output_stream_path,
                          num_users_group1,
                          num_users_group2,
                          events_per_user):
    # 70% churn users

    fake = Faker()

    group1_events_dist = [{'probability': 0.1, 'generator': new_purchase},
                          {'probability': 0.89, 'generator': new_bet},
                          {'probability': 0.01, 'generator': new_win}]

    group2_events_dist = [{'probability': 0.1, 'generator': new_purchase},
                          {'probability': 0.85, 'generator': new_bet},
                          {'probability': 0.05, 'generator': new_win}]

    group1_events = generate_events(fake, num_users_group1, group1_events_dist, events_per_user, True)
    group2_events = generate_events(fake, num_users_group2, group2_events_dist, events_per_user, False)

    events = (group1_events + group2_events)
    events.sort(key=lambda event: event.get('event_time'))

    # ingest events to stream
    batch_size = 1000
    for i in range(0, len(events), batch_size):
        # Convert the events to records
        records = [{'data': json.dumps(event, default=str)} for event in events[i:i+batch_size]]
        v3io_client.batch.stream.put_records(container=container, stream_path=output_stream_path, records=records)

    return v3io_client.batch.wait()


def create_enrichment_table(v3io_client, container, enrichment_table_path):
    for postcode in range(10000, 99999):
        remainder = postcode % 3
        if remainder == 0:
            idx = randint(3, 5)
        elif remainder == 1:
            idx = randint(1, 3)
        else:
            idx = randint(5, 7)

        attr = {'postcode': postcode, 'socioeconomic_idx': idx}
        v3io_client.batch.kv.put(container=container,
                                 table_path=enrichment_table_path,
                                 key=str(postcode),
                                 attributes=attr)
    return v3io_client.batch.wait()


def main(context: MLClientCtx,
         container: str,
         output_stream_path: str,
         enrichment_table_path: str,
         num_users_group1: int = 1400,
         num_users_group2: int = 600,
         events_per_user: int = 1000) -> None:

    v3io_client = v3io.dataplane.Client(endpoint=os.getenv('V3IO_API'),
                                        access_key=os.getenv('V3IO_ACCESS_KEY'))

    stream_resps = generate_event_stream(v3io_client, container, output_stream_path,
                                         num_users_group1, num_users_group2, events_per_user)

    records_sent = sum(len(json.loads(resp.body)['Records']) for resp in stream_resps)
    context.logger.info(f'Records sent {records_sent}')

    failed_records = sum(json.loads(resp.body)['FailedRecordCount'] for resp in stream_resps)

    if failed_records > 0:
        context.logger.warn(f'Failed to stream {failed_records}')
    else:
        context.logger.info('All data streamed successfully.')

    table_resps = create_enrichment_table(v3io_client, container, enrichment_table_path)
    written_items = sum(int(resp.status_code == 200) for resp in table_resps)
    context.logger.info(f'Created enrichment table with {written_items} items')

    pass
