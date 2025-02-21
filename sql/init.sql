\c draslovka;

CREATE TABLE IF NOT EXISTS measurements (
    id SERIAL PRIMARY KEY,
    heatid INT NOT NULL,
    datetime TIMESTAMP NOT NULL,
    datetime_corrected TIMESTAMP NOT NULL,
    end_t NUMERIC NULL,
    gas1 NUMERIC NOT NULL
);