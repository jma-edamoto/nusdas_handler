use nusdas::decode_to_json;

pub fn main() -> Result<(), Box<dyn std::error::Error>>{
  println!("{:?}",decode_to_json("202210200900")?);
  Ok(())
}