use std::{env, fs};
use std::fs::File;
use std::io::prelude::*;

use nom::bytes::complete::{take, tag};
use nom::multi::{count, many0};
use nom::number::complete::{be_f64, be_u16, be_u32, be_u64, be_f32};
use nom::combinator::map;
use nom::{IResult, Finish};

#[derive(Debug)]
struct NUSD{
  creator: String,
  file_size: u64,
  file_version: u32,
  record_size: u32,
  info_record_size: u32,
  subc_record_size: u32,
}

#[derive(Debug)]
struct CNTL{
  difinition_type: String,
  base_time_str: String,
  base_time: u32,
  time_unit: String,
  projection: String,
  x_size :u32,
  y_size :u32,
  basepoint_x :f32,
  basepoint_y :f32,
  basepoint_lat :f32,
  basepoint_lon :f32,
  distance_x :f32,
  distance_y :f32,
  standard_lat_1 :f32,
  standard_lon_1 :f32,
  standard_lat_2 :f32,
  standard_lon_2 :f32,
  other_lat_1 :f32,
  other_lon_1 :f32,
  other_lat_2 :f32,
  other_lon_2 :f32,
  value: String,
  members: Vec<String>,
  forecast_times_1: Vec<u32>,
  forecast_times_2: Vec<u32>,
  planes_1: Vec<String>,
  planes_2: Vec<String>,
  elements: Vec<String>,
}

impl CNTL{
  fn data_size(&self) -> usize{
    self.members.len() * self.forecast_times_1.len()* self.planes_1.len() * self.elements.len()
  }
}

#[derive(Debug)]
struct INDY{
  record_position: Vec<u64>,
  record_length: Vec<u32>,
  record_size: Vec<u32>,
}

struct SUBC{}

#[derive(Debug)]
struct DATA{
  member: String,
  time_1: u32,
  time_2: u32,
  plane_1: String,
  plane_2: String,
  element: String,
  x_size: u32,
  y_size: u32,
  packing: String,
  missing: String,
  data: Vec<f32>,
}

struct INFO{}


#[derive(Debug)]
struct END{
  file_size: u32,
  record_size: u32,
}

struct NusdasData {
  nusd: NUSD,
  cntl: CNTL,
  indy: INDY,
  subc: Vec<SUBC>,
  data: Vec<DATA>,
  info: Vec<INFO>
}

fn read_file(path: &str) -> Result<Vec<u8>,Box<dyn std::error::Error>>{
  let mut file = File::open(path)?;
  let mut buf: Vec<u8> = Vec::new();
  file.read_to_end(&mut buf)?;
  Ok(buf)
}


pub fn load() -> Result<(),Box<dyn std::error::Error>>{
  let input = read_file("202210121800")?;
  let input: &[u8] = &input;
  let (input,nusd) = parse_nusd(input).unwrap();
  let (input, cntl) = parse_ctrl(input).unwrap();
  let (input, indy) = parse_indy(input, cntl.data_size()).unwrap();
  let (input, subc) = parse_subc(input).unwrap();
  let (input, data) = many0(parse_data)(input).unwrap();
  let (input, end) = parse_end(input).unwrap();
  println!("{:?}", end);
  Ok(())
}

fn parse_header<'a>(header: &'a str) -> impl for <'b> Fn(&'b [u8])-> IResult<&'b [u8], usize> + 'a{
  move |input: &[u8]|{
    let (input, record_size) = be_u32(input)?;
    let (input, _) = tag(header)(input)?;
    let (input,_) = be_u32(input)?;
    let (input,_) = take(4usize)(input)?;
    Ok((input, (record_size - 12) as usize))
  }
}

fn take_string(count: usize) -> impl for <'a> Fn(&'a [u8])-> IResult<&'a [u8], String>{
  move |input: &[u8]|{
    let (input, str) = take(count)(input)?;
    let str = String::from_utf8(str.to_vec()).unwrap().trim().to_string();
    Ok((input, str))
  }
}

fn parse_nusd(input: &[u8]) -> IResult<&[u8], NUSD>{
  let (input, block_size) = parse_header("NUSD")(input)?;
  let (input,creator) = take_string(72usize)(input)?;
  let (input, file_size) = be_u64(input)?;
  let (input, file_version) = be_u32(input)?;
  let (input, _) = take(4usize)(input)?;
  let (input, record_size) = be_u32(input)?;
  let (input, info_record_size) = be_u32(input)?;
  let (input, subc_record_size) = be_u32(input)?;
  let (input, _) = take(block_size - 100usize)(input)?;
  let (input, _) = take(4usize)(input)?;
  Ok((input, NUSD{
    creator,
    file_size,
    file_version,
    record_size,
    info_record_size,
    subc_record_size,
  }))
}

fn parse_ctrl(input: &[u8]) -> IResult<&[u8], CNTL>{
  let (input, block_size) = parse_header("CNTL")(input)?;
  let (input, difinition_type) = take_string(16usize)(input)?;
  let (input, base_time_str) = take_string(12usize)(input)?;
  let (input, base_time) = be_u32(input)?;
  let (input, time_unit) = take_string(4usize)(input)?;
  let (input, member_size) = be_u32(input)?;
  let (input, forecast_time_size) = be_u32(input)?;
  let (input, plane_size) = be_u32(input)?;
  let (input, element_size) = be_u32(input)?;
  let (input, projection) = take_string(4usize)(input)?;
  let (input, x_size) = be_u32(input)?;
  let (input, y_size) = be_u32(input)?;
  let (input, basepoint_x) = be_f32(input)?;
  let (input, basepoint_y) = be_f32(input)?;
  let (input, basepoint_lat) = be_f32(input)?;
  let (input, basepoint_lon) = be_f32(input)?;
  let (input, distance_x) = be_f32(input)?;
  let (input, distance_y) = be_f32(input)?;
  let (input, standard_lat_1) = be_f32(input)?;
  let (input, standard_lon_1) = be_f32(input)?;
  let (input, standard_lat_2) = be_f32(input)?;
  let (input, standard_lon_2) = be_f32(input)?;
  let (input, other_lat_1) = be_f32(input)?;
  let (input, other_lon_1) = be_f32(input)?;
  let (input, other_lat_2) = be_f32(input)?;
  let (input, other_lon_2) = be_f32(input)?;
  let (input, value) = take_string(4usize)(input)?;
  let (input, _) = take((4 * (2 + 6)) as usize)(input)?;
  let (input, members) = count(take_string(4usize), member_size as usize)(input)?;
  let (input, forecast_times_1) = count(be_u32,  forecast_time_size as usize)(input)?;
  let (input, forecast_times_2) = count(be_u32,  forecast_time_size as usize)(input)?;
  let (input, planes_1) = count(take_string(6usize),  plane_size as usize)(input)?;
  let (input, planes_2) = count(take_string(6usize),  plane_size as usize)(input)?;
  let (input, elements) = count(take_string(6usize),  element_size as usize)(input)?;
  let gap_size 
    = block_size - (156 + 4 * member_size + 4 * forecast_time_size * 2 + 6 * plane_size * 2 + 6 * element_size) as usize;
  let (input, _) = take(gap_size)(input)?;
  let (input, _) = take(4usize)(input)?;
  Ok((input, CNTL{
    difinition_type,
    base_time_str,
    base_time,
    time_unit,
    projection,
    x_size,
    y_size,
    basepoint_x,
    basepoint_y,
    basepoint_lat,
    basepoint_lon,
    distance_x,
    distance_y,
    standard_lat_1,
    standard_lon_1,
    standard_lat_2,
    standard_lon_2,
    other_lat_1,
    other_lon_1,
    other_lat_2,
    other_lon_2,
    value,
    members,
    forecast_times_1,
    forecast_times_2,
    planes_1,
    planes_2,
    elements,
  }))
}

fn parse_indy(input: &[u8], data_size: usize) -> IResult<&[u8], INDY>{
  let (input, block_size) = parse_header("INDY")(input)?;
  let (input, record_position) = count(be_u64, data_size)(input)?;
  let (input, record_length) = count(be_u32, data_size)(input)?;
  let (input, record_size) = count(be_u32, data_size)(input)?;
  let gap_size = block_size - (8 * data_size + 4 * data_size + 4 * data_size)  as usize;
  let (input, _) = take(gap_size)(input)?;
  let (input, _) = take(4usize)(input)?;
  Ok((input,INDY{
    record_position,
    record_length,
    record_size,
  }))
}

fn parse_subc(input: &[u8]) -> IResult<&[u8], SUBC>{
  Ok((input, SUBC{}))
}

fn parse_data(input: &[u8]) -> IResult<&[u8], DATA>{
  let (input, block_size) = parse_header("DATA")(input)?;
  let (input, member) = take_string(4usize)(input)?;
  let (input, time_1) =be_u32(input)?;
  let (input, time_2) =be_u32(input)?;
  let (input, plane_1) = take_string(6usize)(input)?;
  let (input, plane_2) = take_string(6usize)(input)?;
  let (input, element) = take_string(6usize)(input)?;
  let (input, _) = take(2usize)(input)?;
  let (input, x_size) = be_u32(input)?;
  let (input, y_size) = be_u32(input)?;
  let (input, packing) = take_string(4usize)(input)?;
  let (input, missing) = take_string(4usize)(input)?;
  let (input, base) = be_f32(input)?;
  let (input, amp) = be_f32(input)?;
  let (input, data) = count(
    map(be_u16, |n| amp * (n as f32) + base ),
    (x_size * y_size) as usize
  )(input)?;
  let gap_size = block_size - (48 + 8 + 2 * x_size *  y_size) as usize;
  let (input, _) = take(gap_size)(input)?;
  let (input, _) = take(4usize)(input)?;

  Ok((input, DATA{
    member,
    time_1,
    time_2,
    plane_1,
    plane_2,
    element,
    x_size,
    y_size,
    packing,
    missing,
    data,
  }))
}

fn parse_end(input: &[u8]) -> IResult<&[u8], END>{
  let (input, block_size) = parse_header("END ")(input)?;
  let (input, file_size) = be_u32(input)?;
  let (input, record_size) = be_u32(input)?;
  let (input, _) = take(block_size - 8usize)(input)?;
  let (input, _ ) = take(4usize)(input)?;
  Ok((input, END{
    file_size,
    record_size,
  }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        load();
    }
}
