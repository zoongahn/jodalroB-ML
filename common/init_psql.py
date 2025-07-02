import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sshtunnel import SSHTunnelForwarder
import socket

load_dotenv()

POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))

# Build SQLAlchemy database URL
DATABASE_URL = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)


def connect_psql_local():
	# Create SQLAlchemy engine with TCP keepalives
	engine = create_engine(
		DATABASE_URL,
		echo=False,
		pool_pre_ping=True,
		connect_args={
			"keepalives": 1,
			"keepalives_idle": 60,
			"keepalives_interval": 10,
			"keepalives_count": 10
		}
	)
	# Create session factory and open session
	SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
	session = SessionLocal()

	return None, session


def connect_psql_via_ssh():
	SSH_HOST = os.getenv("SSH_HOST")
	SSH_PORT = int(os.getenv("SSH_PORT"))
	SSH_USERNAME = os.getenv("SSH_USER")
	SSH_PASSWORD = None  # 비밀번호 방식 사용 시 설정
	SSH_PKEY = os.getenv("SSH_PKEY_PATH")  # SSH 키 사용

	# Start SSH tunnel
	server = SSHTunnelForwarder(
		(SSH_HOST, SSH_PORT),
		ssh_username=SSH_USERNAME,
		ssh_password=SSH_PASSWORD if SSH_PKEY is None else None,
		ssh_pkey=SSH_PKEY if SSH_PKEY else None,
		remote_bind_address=(POSTGRES_HOST, POSTGRES_PORT),
	)
	server.start()
	local_port = server.local_bind_port

	# Create engine pointing to local end of tunnel
	url = (
		f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
		f"@127.0.0.1:{local_port}/{POSTGRES_DB}"
	)
	engine = create_engine(
		url,
		echo=False,
		pool_pre_ping=True
	)
	SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
	session = SessionLocal()
	return server, session


def init_psql():
	server, session = None, None
	ENV = os.getenv("ENV")

	if ENV == "remote":
		server, session = connect_psql_via_ssh()
	elif ENV == "local":
		server, session = connect_psql_local()

	return server, session
