from sqlalchemy import create_engine, Column, Integer, String,Float
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = (
    "mssql+pyodbc://----/TumorTahminDB?"
    "driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
)

Base = declarative_base()

# BreakHis tablosu
class BreakhisPrediction(Base):
    __tablename__ = 'breakhis_predictions'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    surname = Column(String)
    image_path = Column(String)
    prediction = Column(String)
    confidence = Column(Float)

class BrainPrediction(Base):
    __tablename__ = 'brain_predictions'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    surname = Column(String)
    image_path = Column(String)
    prediction = Column(String)
    confidence = Column(Float)


engine = create_engine(DATABASE_URL)
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(bind=engine)
