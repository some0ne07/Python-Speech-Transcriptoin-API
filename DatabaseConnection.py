from pymongo import MongoClient
from gridfs import GridFS
from bson import ObjectId
from constants import connection_string

# Connect to MongoDB using the provided connection string
client = MongoClient(connection_string)

# Access the "VideoDb" database
database = client["VideoDb"]
bucket_name = "file"

# Access the "TranscriptDb" database
TextDb = client["TranscriptDb"]
transcriptCollection = TextDb["transcript"]

# Create a GridFS instance for file storage
fs = GridFS(database, collection=bucket_name)

def retrieve_mp4_from_gridfs(path: str):
    try:
        object_id = ObjectId(path)
    except Exception:
        return None

    # Find the file object in GridFS based on the metadata fileId
    file_object = fs.find_one({"metadata.fileId": object_id})

    if file_object:
        file_content = file_object.read()

        if file_content:
            # Write the file content to the local file
            with open("/content/file.mp4", "wb") as file:
                file.write(file_content)
            return '/content/file.mp4'

    return None

def insert_transcript(path: str, transcript: str):
    doc = {
        "fileid": ObjectId(path),
        "transcript": transcript
    }
    # Insert the document into the transcript collection
    transcriptCollection.insert_one(doc)