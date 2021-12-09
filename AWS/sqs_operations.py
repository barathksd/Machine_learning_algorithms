
import boto3

# Create SQS client
sqs = boto3.client('sqs')
l = sqs.list_queues()['QueueUrls']

def send_message(queue_url,body,attributes=None,delay=10):
    # Send message to SQS queue
    response = sqs.send_message(
        QueueUrl=queue_url,
        DelaySeconds=delay,
        MessageAttributes=attributes,
        MessageBody=body
    )
    return response

def delete_message(queue_url,rh):
    
    response = sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=rh)
    return response



if __name__ == '__main__':
    
    queue_url = [i for i in l if 'pipeline' in i][0]
    
    #queue_url = 'SQS_QUEUE_URL'
    attributes = {
            'Title': {
                'DataType': 'String',
                'StringValue': 'The Whistler'
            },
            'Author': {
                'DataType': 'String',
                'StringValue': 'John Grisham'
            },
            'WeeksOn': {
                'DataType': 'Number',
                'StringValue': '6'
            }
        }
    
    
    body = (
            'Information about current NY Times fiction bestseller for '
            'week of 12/11/2016.'
        )
    
    response = send_message(queue_url, body, attributes)
    # rh = response['Messages'][0]['ReceiptHandle']

